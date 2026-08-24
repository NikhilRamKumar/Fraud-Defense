import os
import io
import time
import uuid
import joblib
import pandas as pd
import qrcode
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse

app = FastAPI(title="KavachQR Fraud Defense Gateway")

# -------------------------------------------------------------------------
# 1. Load ML Artifacts
# -------------------------------------------------------------------------
MODEL_PATH = "models/lgbm_fraud_model.pkl"
CONFIG_PATH = "models/model_config.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)
    OPTIMAL_THRESHOLD = config.get("threshold", 0.5)
else:
    model = None
    OPTIMAL_THRESHOLD = 0.5

# In-Memory Session Storage for Demo
# Schema: { session_id: { "created_at": float, "status": str, "ip": str, ... } }
active_sessions = {}

# -------------------------------------------------------------------------
# 2. Endpoint: Generate Dynamic QR Code
# -------------------------------------------------------------------------
@app.get("/generate_qr")
def generate_qr(request: Request, merchant_id: str = "MERCHANT_001", amount: float = 100.0):
    """Generates a dynamic session token and returns a QR Code image encoding an API router URL."""
    session_id = f"SESS_{uuid.uuid4().hex[:8].upper()}"
    
    # Store session state
    active_sessions[session_id] = {
        "created_at": time.time(),
        "merchant_id": merchant_id,
        "amount": amount,
        "bank_webhook_received": 0,
        "status": "PENDING", # PENDING, VERIFIED, SPOOF_BLOCKED
        "ip_requests": {}
    }

    # Public gateway routing URL (simulated for local testing)
    host = request.headers.get("host", "127.0.0.1:8000")
    gateway_url = f"http://{host}/scan/{session_id}"

    # Generate QR Code image in memory
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(gateway_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf)
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png", headers={"X-Session-ID": session_id})

# -------------------------------------------------------------------------
# 3. Endpoint: Handles Scan & Runs ML Spoof Classifier
# -------------------------------------------------------------------------
@app.get("/scan/{session_id}")
def handle_scan(session_id: str, request: Request):
    """When scanned, extracts metadata, passes to ML model, and handles routing."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session Expired or Invalid")

    session = active_sessions[session_id]
    scan_time_delta = time.time() - session["created_at"]
    client_ip = request.client.host

    # Track request velocity per IP
    session["ip_requests"][client_ip] = session["ip_requests"].get(client_ip, 0) + 1
    ip_velocity = session["ip_requests"][client_ip]

    # Simple User-Agent heuristic
    user_agent = request.headers.get("user-agent", "").lower()
    user_agent_score = 0.30 if ("python" in user_agent or "curl" in user_agent) else 0.95

    # Prepare features for LightGBM model
    features = pd.DataFrame([{
        'scan_time_delta': scan_time_delta,
        'ip_request_velocity': ip_velocity,
        'user_agent_score': user_agent_score,
        'bank_webhook_received': session["bank_webhook_received"]
    }])

    # Predict Spoof Probability
    if model:
        spoof_prob = model.predict_proba(features)[0][1]
        is_spoof = spoof_prob >= OPTIMAL_THRESHOLD
    else:
        is_spoof = scan_time_delta > 60 or ip_velocity > 5

    if is_spoof:
        session["status"] = "SPOOF_BLOCKED"
        return JSONResponse(
            status_code=403,
            content={
                "error": "SECURITY_ALERT_SPOOF_BLOCKED",
                "message": "Offline payment spoofing attempt flagged by KavachQR Engine.",
                "session_id": session_id
            }
        )

    # Legitimate Attempt: Redirect to standard UPI Deep Link
    merchant_id = session["merchant_id"]
    amount = session["amount"]
    upi_intent = f"upi://pay?pa={merchant_id}@okaxis&pn=KiranaStore&am={amount}&tr={session_id}&cu=INR"
    
    return RedirectResponse(url=upi_intent, status_code=302)

# -------------------------------------------------------------------------
# 4. Endpoint: Bank Webhook Simulation
# -------------------------------------------------------------------------
@app.post("/mock_bank_webhook")
def mock_bank_webhook(payload: dict):
    """Simulates an acquiring bank sending a ledger settlement webhook."""
    session_id = payload.get("session_id")
    status = payload.get("status") # "SUCCESS" or "FAILED"

    if session_id in active_sessions and status == "SUCCESS":
        active_sessions[session_id]["bank_webhook_received"] = 1
        active_sessions[session_id]["status"] = "VERIFIED"
        return {"status": "SETTLED", "session_id": session_id}
    
    return {"status": "REJECTED", "message": "Invalid transaction state"}

# -------------------------------------------------------------------------
# 5. Endpoint: Merchant Dashboard Status Polling
# -------------------------------------------------------------------------
@app.get("/merchant_status/{session_id}")
def merchant_status(session_id: str):
    """Polled by the merchant counter screen to show status."""
    if session_id not in active_sessions:
        return {"status": "EXPIRED"}
    
    return {
        "session_id": session_id,
        "status": active_sessions[session_id]["status"],
        "bank_webhook_received": active_sessions[session_id]["bank_webhook_received"]
    }