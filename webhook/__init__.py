import os
import json
import time
from typing import List, Dict, Any, Optional

import azure.functions as func
import httpx
from openai import OpenAI

# ------- Environment Variables -------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Keep replies concise.")
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "8"))  # number of messages to keep (user+assistant)
MEMORY_TTL_SEC = int(os.getenv("MEMORY_TTL_SEC", "86400"))  # 24h ttl for in-memory cache

# Optional: persistent session store
AZURE_TABLES_CONNECTION_STRING = os.getenv("AZURE_TABLES_CONNECTION_STRING")
AZURE_TABLES_TABLE_NAME = os.getenv("AZURE_TABLES_TABLE_NAME", "TelegramChatMemory")

TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

COMMAND_RESPONSES = {
    "/start": "Hi! I'm your assistant. Ask me anything to get started.",
    "/help": "Send me a message and I'll do my best to help. Commands: /start, /help, /reset.",
    "/reset": "Cleared the recent chat memory. We can start fresh!",
}

# OpenAI client (Azure endpoint)
client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
)

# ------- Session Memory Layer (pluggable) -------

class BaseMemoryStore:
    def get(self, chat_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError
    def put(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        raise NotImplementedError
    def delete(self, chat_id: str) -> None:
        raise NotImplementedError

class InMemoryStore(BaseMemoryStore):
    """Ephemeral per-process cache with TTL"""
    def __init__(self, ttl_sec: int = 86400):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_sec

    def get(self, chat_id: str) -> List[Dict[str, Any]]:
        rec = self._store.get(chat_id)
        now = time.time()
        if not rec:
            return []
        if rec["exp"] < now:
            del self._store[chat_id]
            return []
        return rec["messages"]

    def put(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        self._store[chat_id] = {"messages": messages, "exp": time.time() + self._ttl}

    def delete(self, chat_id: str) -> None:
        self._store.pop(chat_id, None)

class TableStore(BaseMemoryStore):
    """Azure Table Storage backed store (survives restarts/scale)"""
    def __init__(self, conn_str: str, table_name: str):
        from azure.data.tables import TableServiceClient  # lazy import
        svc = TableServiceClient.from_connection_string(conn_str)
        self.table = svc.create_table_if_not_exists(table_name=table_name)

    def _key(self, chat_id: str):
        return ("chat", str(chat_id))  # PartitionKey, RowKey

    def get(self, chat_id: str) -> List[Dict[str, Any]]:
        pk, rk = self._key(chat_id)
        try:
            entity = self.table.get_entity(partition_key=pk, row_key=rk)
            payload = entity.get("messagesJson") or "[]"
            return json.loads(payload)
        except Exception:
            return []

    def put(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        pk, rk = self._key(chat_id)
        entity = {
            "PartitionKey": pk,
            "RowKey": rk,
            "messagesJson": json.dumps(messages),
            "ts": int(time.time()),
        }
        # upsert to handle first write / subsequent updates
        self.table.upsert_entity(mode="merge", entity=entity)

    def delete(self, chat_id: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        pk, rk = self._key(chat_id)
        try:
            self.table.delete_entity(partition_key=pk, row_key=rk)
        except ResourceNotFoundError:
            return
        except Exception as exc:
            print(f"[warn] memory_store.delete failed: {exc}")

# Select store
if AZURE_TABLES_CONNECTION_STRING:
    try:
        memory_store: BaseMemoryStore = TableStore(AZURE_TABLES_CONNECTION_STRING, AZURE_TABLES_TABLE_NAME)
    except Exception as e:
        print(f"[warn] TableStore init failed, falling back to InMemoryStore: {e}")
        memory_store = InMemoryStore(ttl_sec=MEMORY_TTL_SEC)
else:
    memory_store = InMemoryStore(ttl_sec=MEMORY_TTL_SEC)

# ------- Helpers -------

def build_messages(history: List[Dict[str, Any]], user_text: str) -> List[Dict[str, str]]:
    """Combine system prompt, history, and current user turn."""
    base = [{"role": "system", "content": SYSTEM_PROMPT}]
    return base + history + [{"role": "user", "content": user_text}]

def clip_history(history: List[Dict[str, Any]], max_turns: int) -> List[Dict[str, Any]]:
    """Keep last N messages (user+assistant)."""
    if max_turns <= 0:
        return []
    return history[-max_turns:]

def extract_bot_command(message: Dict[str, Any]) -> Optional[str]:
    """Return the lower-cased bot command (e.g. '/start') or None."""
    text = message.get("text") or ""
    entities = message.get("entities") or []

    for entity in entities:
        if entity.get("type") == "bot_command" and entity.get("offset", 0) == 0:
            length = entity.get("length", 0)
            cmd = text[:length]
            return cmd.split("@", 1)[0].lower()

    first_token = text.split()[0].lower() if text else ""
    if first_token.startswith("/"):
        return first_token.split("@", 1)[0]
    return None

async def send_telegram_message(chat_id: int, text: str):
    text = text[:4096]  # Telegram hard cap
    async with httpx.AsyncClient(timeout=10) as http:
        await http.post(f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": text})

# ------- Azure Function Entrypoint -------

async def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method != "POST":
        return func.HttpResponse("OK", status_code=200)

    try:
        update = req.get_json()
    except Exception:
        return func.HttpResponse("Invalid JSON", status_code=400)

    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return func.HttpResponse("OK", status_code=200)

    chat_id = msg["chat"]["id"]
    user_text = msg["text"]

    chat_key = str(chat_id)

    command = extract_bot_command(msg)
    if command:
        if command == "/reset":
            try:
                memory_store.delete(chat_key)
            except Exception as exc:
                print(f"[warn] memory_store.delete raised: {exc}")
        reply = COMMAND_RESPONSES.get(command, "Sorry, I don't recognize that command.")
        try:
            await send_telegram_message(chat_id, reply)
        except Exception as e:
            print(f"[warn] send_telegram_message failed for command {command}: {e}")
        return func.HttpResponse("OK", status_code=200)

    # 1) Load existing history for this chat
    history = memory_store.get(chat_key)  # list of {"role": "...", "content": "..."}

    # 2) Build request with system + history + new user turn
    messages = build_messages(history, user_text)

    # 3) Call Azure OpenAI
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.4,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Error: {type(e).__name__}"

    # 4) Update and persist history (append user & assistant turns)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    history = clip_history(history, MEMORY_MAX_TURNS)
    try:
        memory_store.put(chat_key, history)
    except Exception as e:
        print(f"[warn] memory_store.put failed: {e}")

    # 5) Reply to Telegram
    try:
        await send_telegram_message(chat_id, answer)
    except Exception as e:
        print(f"[warn] send_telegram_message failed: {e}")

    return func.HttpResponse("OK", status_code=200)
