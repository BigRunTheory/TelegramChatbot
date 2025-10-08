import os, json
import azure.functions as func
import httpx
from openai import OpenAI

# --- Environment variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

# --- API clients ---
TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method != "POST":
        return func.HttpResponse("OK", status_code=200)

    try:
        update = req.get_json()
    except Exception:
        return func.HttpResponse("Bad request", status_code=400)

    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return func.HttpResponse("OK", status_code=200)

    chat_id = msg["chat"]["id"]
    user_text = msg["text"]

    # Call Azure OpenAI
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.4,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Error: {type(e).__name__}"

    # Reply to user via Telegram
    async with httpx.AsyncClient(timeout=10) as http:
        await http.post(f"{TG_API}/sendMessage",
                        json={"chat_id": chat_id, "text": answer[:4096]})

    return func.HttpResponse("OK", status_code=200)
