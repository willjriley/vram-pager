"""Execute code on RunPod via Jupyter WebSocket API."""
import os
import requests
import json
import time
import websocket

POD_ID = os.environ.get("RUNPOD_POD_ID", "")
TOKEN = "dev"
JUPYTER_URL = f"https://{POD_ID}-8888.proxy.runpod.net"

def get_session():
    s = requests.Session()
    s.headers.update({"Authorization": f"token {TOKEN}"})
    return s

def run(code, timeout=300):
    session = get_session()
    # Get or create kernel
    kernels = session.get(f"{JUPYTER_URL}/api/kernels?token={TOKEN}").json()
    if isinstance(kernels, list) and len(kernels) > 0:
        kid = kernels[0]["id"]
    else:
        kid = session.post(f"{JUPYTER_URL}/api/kernels?token={TOKEN}").json()["id"]

    ws_url = JUPYTER_URL.replace("https://", "wss://") + f"/api/kernels/{kid}/channels?token={TOKEN}"
    ws = websocket.create_connection(ws_url, timeout=30)

    mid = f"run_{time.time()}"
    msg = {
        "header": {"msg_id": mid, "msg_type": "execute_request", "username": "", "session": "", "date": "", "version": "5.3"},
        "parent_header": {},
        "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "buffers": [],
        "channel": "shell"
    }
    ws.send(json.dumps(msg))

    output = ""
    start = time.time()
    while time.time() - start < timeout:
        try:
            ws.settimeout(15)
            r = json.loads(ws.recv())
            if r.get("parent_header", {}).get("msg_id") != mid:
                continue
            if r.get("msg_type") == "stream":
                output += r["content"]["text"]
                print(r["content"]["text"], end="", flush=True)
            elif r.get("msg_type") == "error":
                err = r["content"]
                output += f"ERROR: {err.get('ename','')}: {err.get('evalue','')}\n"
                print(f"ERROR: {err.get('ename','')}: {err.get('evalue','')}")
            elif r.get("msg_type") == "execute_reply":
                break
        except websocket.WebSocketTimeoutException:
            continue
    ws.close()
    return output

if __name__ == "__main__":
    print(run("import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); import subprocess; r=subprocess.run(['nvcc','--version'],capture_output=True,text=True); print(r.stdout)"))
