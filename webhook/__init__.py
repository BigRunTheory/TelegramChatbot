import os
import json
import logging

# ---------- Env ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
ECHO_ONLY = os.getenv("ECHO_ONLY", "false").lower() == "true"

# For safety if token missing
TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None


async def main(req):
    """Azure Function entrypoint with lazy imports + rich logging."""
    logging.info("ENTER main()")

    # ---- Lazy imports so missing deps become visible in logs (not silent 500) ----
    try:
        import azure.functions as func
        import httpx
        from openai import OpenAI
        logging.info("Imports OK")
    except Exception:
        logging.exception("Import failed (likely missing package: azure-functions/httpx/openai)")
        # Return 200 so Telegram stops retrying while you fix deps
        return "Import error"

    # ---- Parse request body safely ----
    try:
        update = req.get_json()
        logging.info("Incoming update: %s", json.dumps(update)[:800])
    except Exception:
        logging.exception("Bad JSON")
        return func.HttpResponse("Bad Request", status_code=400)

    if req.method != "POST":
        logging.info("Non-POST request; returning 200")
        return func.HttpResponse("OK", status_code=200)

    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        logging.info("No text message in update; returning 200")
        return func.HttpResponse("OK", status_code=200)

    chat_id = msg["chat"]["id"]
    user_text = msg["text"]
    logging.info("CHAT %s TEXT %s", chat_id, user_text[:120])

    # ---- Build reply ----
    answer = "(no content)"
    if ECHO_ONLY:
        answer = f"(echo) {user_text}"
        logging.info("ECHO_ONLY enabled -> %s", answer[:120])
    else:
        if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT):
            logging.error("Azure OpenAI env missing. endpoint=%s deployment=%s key_set=%s",
                          AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, bool(AZURE_OPENAI_API_KEY))
            answer = "Sorry—server config for Azure OpenAI is incomplete."
        else:
            try:
                client = OpenAI(
                    api_key=AZURE_OPENAI_API_KEY,
                    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
                )
                resp = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,   # deployment name, not raw model id
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.4,
                )
                answer = resp.choices[0].message.content or "(empty)"
                logging.info("OpenAI OK (len=%d)", len(answer))
            except Exception:
                logging.exception("OpenAI call failed")
                answer = "Sorry—there was an issue talking to Azure OpenAI."

    # ---- Send reply to Telegram ----
    if not TG_API:
        logging.error("TELEGRAM_BOT_TOKEN missing; cannot sendMessage")
    else:
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                r = await http.post(f"{TG_API}/sendMessage",
                                    json={"chat_id": chat_id, "text": answer[:4096]})
            logging.info("Telegram sendMessage status=%s body=%s",
                         r.status_code, (r.text or "")[:300])
        except Exception:
            logging.exception("sendMessage failed")

    # ---- Always 200 to stop Telegram retries ----
    return func.HttpResponse("OK", status_code=200)
