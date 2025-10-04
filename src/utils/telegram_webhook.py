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

# ✅ New endpoint: manually trigger the flow
@app.post("/webhook/runflow")
async def runflow_endpoint():
    try:
        # Run your script with the exact environment
        subprocess.Popen([
            "/root/selenium-env/bin/python",
            "/root/analytical_services/src/analysis/stats_sport_bet/alert_scan_match_use_for_tele.py"
        ])
        return {"ok": True, "message": "Flow started successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ✅ Telegram webhook (still works for chat commands)
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("📩 Incoming update:", data)

    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if not text:
        return {"ok": True}

    if text.startswith("/runflow alert_scan_match_use_for_tele"):
        try:
            subprocess.Popen([
                "/root/selenium-env/bin/python",
                "/root/analytical_services/src/analysis/stats_sport_bet/alert_scan_match_use_for_tele.py"
            ])
            send_message(chat_id, "✅ Flow 'alert_scan_match' started!")
        except Exception as e:
            send_message(chat_id, f"❌ Error: {e}")
    
    elif text.startswith("/runflow"):
        parts = text.split(" ", 1)
        if len(parts) == 2:
            team_name = parts[1].strip()
        else:
            team_name = None
            
        cmd = [
            "/root/selenium-env/bin/python",
            "/root/analytical_services/src/analysis/stats_sport_bet/alert_scan_match_search_bet.py"
        ]
        if team_name:
            cmd.append(team_name)

        try:
            subprocess.Popen(cmd)
        except Exception as e:
            send_message(chat_id, f"❌ Error: {e}")

        if team_name:
            send_message(chat_id, f"✅ Flow started for team '{team_name}'")
        else:
            send_message(chat_id, "✅ Flow started for ALL teams")



    elif text.startswith("/help"):
        send_message(chat_id, "Commands:\n/runflow alert_scan_match - Start the flow")

    return {"ok": True}


# ✅ Send Telegram messages
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})
    
    
    



