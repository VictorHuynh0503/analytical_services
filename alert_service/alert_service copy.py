from fastapi import FastAPI
import uvicorn
import requests
import json
import os

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1404056069080219698/EhiyLSwoR-yS_PO9TDVcw83sPjGwXNOvLVpYGYlBJqiiPb1QHhsAHrX_-A6dXDYdTInk"  # Replace
ALERT_KEYS_FILE = "alerted_keys.json"

app = FastAPI()

def load_alerted_keys():
    if os.path.exists(ALERT_KEYS_FILE):
        with open(ALERT_KEYS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_alerted_keys(keys):
    with open(ALERT_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(keys), f)

alerted_keys = load_alerted_keys()

def send_discord_message(content: str):
    payload = {"content": content}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

@app.post("/alert")
def create_alert(entry: dict):
    """
    Example request body:
    {
        "type": "error",
        "id": 123,
        "value": 95
    }
    """
    key = f"{entry.get('type')}|{entry.get('id')}"
    
    if key in alerted_keys:
        return {"status": "ignored", "reason": "duplicate alert"}

    send_discord_message(f"🚨 Alert: {entry}")
    alerted_keys.add(key)
    save_alerted_keys(alerted_keys)

    return {"status": "alert_sent", "entry": entry}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
