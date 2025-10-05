# webhook_server.py
from fastapi import FastAPI, Request
import requests, subprocess, shlex

app = FastAPI()

BOT_TOKEN = "1200942736:AAEG8y9qyJ7aHefUm4vt_xKqkNBxfKd3qCc"
SECRET = "supersecret"  # optional, for verifying Telegram

# Map short names to real scripts
SCRIPTS = {
    "alert_scan_match_search_bet": "/root/analytical_services/src/analysis/stats_sport_bet/alert_scan_match_search_bet.py",
    "alert_scan_match_use_for_tele": "/root/analytical_services/src/analysis/stats_sport_bet/alert_scan_match_use_for_tele.py",
}

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
        "commands": ["/runflow <script_name> [arg]", "/help"]
    }

# ✅ Generic run command
def run_script(script_name: str, arg: str = None):
    if script_name not in SCRIPTS:
        raise ValueError(f"Unknown script: {script_name}")

    cmd = ["/root/selenium-env/bin/python", SCRIPTS[script_name]]
    if arg:
        cmd.append(arg)

    print("▶️ Running command:", shlex.join(cmd))
    subprocess.Popen(cmd)  # non-blocking
    return True

# ✅ API endpoint to trigger run manually
@app.post("/webhook/runflow")
async def runflow_endpoint(script_name: str, arg: str = None):
    try:
        run_script(script_name, arg)
        return {"ok": True, "message": f"Flow '{script_name}' started"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ✅ Telegram webhook (for commands)
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("📩 Incoming update:", data)

    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if not text:
        return {"ok": True}

    if text.startswith("/runflow"):
        parts = text.split()
        command = parts[0].split("@")[0]  # remove @BotName if present
        if command == "/runflow":
            if len(parts) >= 2:
                script_name = parts[1]
                arg = parts[2] if len(parts) > 2 else None
                try:
                    run_script(script_name, arg)
                    msg = f"✅ Flow '{script_name}' started"
                    if arg:
                        msg += f" with argument '{arg}'"
                    send_message(chat_id, msg)
                except Exception as e:
                    send_message(chat_id, f"❌ Error: {e}")
            else:
                send_message(chat_id, "⚠️ Usage: /runflow <script_name> [arg]")

    elif text.startswith("/help"):
        commands = "\n".join([f"/runflow {k} [arg]" for k in SCRIPTS.keys()])
        send_message(chat_id, f"Available commands:\n{commands}")

    return {"ok": True}

# ✅ Send Telegram messages
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})
