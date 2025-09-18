# webhook_server.py
from fastapi import FastAPI, Request
import requests, subprocess

app = FastAPI()

token="1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
chat_id = "@vihuynh_alert"

BOT_TOKEN = "1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
SECRET = "supersecret"  # optional, for verifying Telegram

@app.post("/webhook")
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON"}

    print("📩 Incoming update:", data)

    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if not text:
        return {"ok": True}

    # ✅ Command: /runflow <flow_name>
    if text.startswith("/runflow"):
        parts = text.split(" ", 1)
        if len(parts) == 2:
            flow_name = parts[1].strip()
        else:
            flow_name = "default"

        try:
            # Example: trigger Python script in "flows/" folder
            subprocess.Popen(["python", f"flows/{flow_name}.py"])

            send_message(chat_id, f"✅ Flow '{flow_name}' started!")
        except Exception as e:
            send_message(chat_id, f"❌ Error: {e}")

    # Optional: help command
    elif text.startswith("/help"):
        send_message(chat_id, "Available commands:\n/runflow <flow_name> - Start a flow")

    return {"ok": True}
