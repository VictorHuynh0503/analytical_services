# webhook_server.py
from fastapi import FastAPI, Request
import requests, subprocess

app = FastAPI()

token="1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
chat_id = "@vihuynh_alert"

BOT_TOKEN = "1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
SECRET = "supersecret"  # optional, for verifying Telegram

# ✅ Health check root
@app.get("/")
async def root():
    return {"status": "ok", "message": "FastAPI server running"}

# ✅ Manual help page
@app.get("/webhook/help")
async def webhook_help():
    return {
        "info": "This is the Telegram webhook endpoint.",
        "usage": "POST a JSON payload from Telegram to /webhook",
        "commands": ["/runflow <flow_name>", "/help"]
    }

# ✅ Telegram webhook endpoint (must be POST)
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if not text:
        return {"ok": True}

    # Command: /runflow
    if text.startswith("/runflow"):
        parts = text.split(" ", 1)
        if len(parts) == 2:
            flow_name = parts[1].strip()
        else:
            flow_name = "default"

        try:
            subprocess.Popen(["python", f"flows/{flow_name}.py"])
            send_message(chat_id, f"✅ Flow '{flow_name}' started!")
        except Exception as e:
            send_message(chat_id, f"❌ Error: {e}")

    elif text.startswith("/help"):
        send_message(chat_id, "Available commands:\n/runflow <flow_name> - Start a flow")

    return {"ok": True}


# ✅ Send Telegram messages
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})
