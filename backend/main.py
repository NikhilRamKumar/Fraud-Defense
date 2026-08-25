import os
import io
import time
import uuid
import joblib
import pandas as pd
import qrcode

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------------------------
# KavachQR Multi-Agent Defense Layer
# -------------------------------------------------------------------------
# Agents are advisory/defense-only. They never directly block a real payment.
from agents.bridge import (
    sync_scan,
    run_investigator_gray_zone,
    run_evidence_compiler,
    run_ring_detector,
    run_merchant_copilot,
    run_red_team_sandbox,
)

from agents.tool_implementations import (
    STORE as AGENT_STORE,
    seed_demo_data,
)

# -------------------------------------------------------------------------
# 1. FastAPI Application
# -------------------------------------------------------------------------

app = FastAPI(
    title="KavachQR Dual-Mode Fraud Defense Gateway",
    version="1.1.0",
)

# -------------------------------------------------------------------------
# 2. CORS
# -------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],
)

# -------------------------------------------------------------------------
# 3. Load ML Artifacts
# -------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "lgbm_fraud_model.pkl"
)

CONFIG_PATH = os.path.join(
    BASE_DIR, "models", "model_config.pkl"
)

if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)
    OPTIMAL_THRESHOLD = config.get("threshold", 0.5)

    print("ML model loaded successfully.")
    print(f"Optimal threshold: {OPTIMAL_THRESHOLD:.2f}")
else:
    model = None
    OPTIMAL_THRESHOLD = 0.5

    print("WARNING: ML model artifacts not found.")
    print("Running with fallback detection rules.")

# -------------------------------------------------------------------------
# 4. In-Memory Session Database
# -------------------------------------------------------------------------

active_sessions = {}

# -------------------------------------------------------------------------
# 5. Health Check
# -------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "KavachQR Dual-Mode Fraud Defense Gateway",
        "status": "online",
        "ml_model_loaded": model is not None,
        "threshold": OPTIMAL_THRESHOLD,
        "agents": [
            "investigator",
            "evidence_compiler",
            "ring_detector",
            "copilot",
            "red_team",
        ],
        "agent_policy": "advisory / defense-only",
    }

# -------------------------------------------------------------------------
# 6. Generate Dynamic QR
# -------------------------------------------------------------------------

@app.get("/generate_qr")
def generate_qr(
    request: Request,
    merchant_id: str = "MERCHANT_001",
    amount: float = 0.0,
):
    """
    Generates dynamic session QR.

    amount > 0  -> fixed amount mode
    amount == 0 -> customer enters amount
    """

    session_id = f"SESS_{uuid.uuid4().hex[:8].upper()}"

    active_sessions[session_id] = {
        "created_at": time.time(),
        "merchant_id": merchant_id,

        # Keep BOTH names:
        # expected_amount is used by your current frontend/backend.
        # amount is used by the agent bridge.
        "mode": "FIXED" if amount > 0 else "ZERO_INPUT",
        "expected_amount": amount,
        "amount": amount,

        "settled_amount": 0.0,
        "bank_webhook_received": 0,
        "status": "PENDING",
        "ip_requests": {},

        # Agent metadata
        "agent_status": "not_triggered",
        "investigator_status": None,
    }

    host = request.headers.get(
        "host",
        "127.0.0.1:8000",
    )

    gateway_url = f"http://{host}/scan/{session_id}"

    qr = qrcode.QRCode(
        version=1,
        box_size=8,
        border=2,
    )

    qr.add_data(gateway_url)
    qr.make(fit=True)

    img = qr.make_image(
        fill_color="black",
        back_color="white",
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"X-Session-ID": session_id},
    )

# -------------------------------------------------------------------------
# 7. QR Scan + LightGBM Fraud Gate + Investigator Gray Zone
# -------------------------------------------------------------------------

@app.get("/scan/{session_id}")
def handle_scan(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session Expired or Invalid",
        )

    session = active_sessions[session_id]

    scan_time_delta = time.time() - session["created_at"]

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    session["ip_requests"][client_ip] = (
        session["ip_requests"].get(client_ip, 0) + 1
    )

    ip_velocity = session["ip_requests"][client_ip]

    user_agent = request.headers.get(
        "user-agent",
        "",
    ).lower()

    user_agent_score = (
        0.30
        if ("python" in user_agent or "curl" in user_agent)
        else 0.95
    )

    # Save ML signals into the real FastAPI session.
    session["scan_time_delta"] = scan_time_delta
    session["ip_request_velocity"] = ip_velocity
    session["user_agent_score"] = user_agent_score
    session["optimal_threshold"] = OPTIMAL_THRESHOLD

    features = pd.DataFrame([{
        "scan_time_delta": scan_time_delta,
        "ip_request_velocity": ip_velocity,
        "user_agent_score": user_agent_score,
        "bank_webhook_received": session["bank_webhook_received"],
    }])

    # -------------------------------------------------------------
    # ML prediction
    # -------------------------------------------------------------

    if model is not None:
        spoof_prob = float(
            model.predict_proba(features)[0][1]
        )
        is_spoof = spoof_prob >= OPTIMAL_THRESHOLD

    else:
        # Fallback detection rules.
        is_spoof = (
            scan_time_delta > 60
            or ip_velocity > 5
            or user_agent_score < 0.5
        )

        # IMPORTANT:
        # The agent bridge needs a probability even when the ML model
        # is unavailable.
        spoof_prob = 0.90 if is_spoof else 0.10

    session["classifier_confidence"] = spoof_prob

    # -------------------------------------------------------------
    # Synchronize this real FastAPI session with agent STORE.
    # -------------------------------------------------------------

    sync_scan(
        session_id=session_id,
        session=session,
        ip_address=client_ip,
        user_agent=user_agent,
        spoof_probability=spoof_prob,
        flagged=is_spoof,
    )

    # -------------------------------------------------------------
    # Gray-zone Investigator
    # -------------------------------------------------------------
    # IMPORTANT:
    # Investigator does NOT make the payment decision.
    # LightGBM remains the deterministic gate.

    if 0.35 <= spoof_prob <= 0.65:
        session["agent_status"] = "investigator_queued"

        background_tasks.add_task(
            run_investigator_gray_zone,
            session_id,
            spoof_prob,
        )
    else:
        session["agent_status"] = "not_triggered"

    # -------------------------------------------------------------
    # Real-time fraud block
    # -------------------------------------------------------------

    if is_spoof:
        session["status"] = "SPOOF_BLOCKED"

        # Evidence Compiler can later be triggered through /dispute.
        return JSONResponse(
            status_code=403,
            content={
                "error": "SECURITY_ALERT_SPOOF_BLOCKED",
                "message": "Offline spoof app detected by KavachQR Engine.",
                "session_id": session_id,
                "spoof_probability": round(spoof_prob, 4),
                "agent_status": session["agent_status"],
            },
        )

    # -------------------------------------------------------------
    # Legitimate attempt -> UPI Intent
    # -------------------------------------------------------------

    merchant_id = session["merchant_id"]

    amt_param = (
        f"&am={session['expected_amount']}"
        if session["expected_amount"] > 0
        else ""
    )

    upi_intent = (
        f"upi://pay?"
        f"pa={merchant_id}@okaxis"
        f"&pn=KiranaStore"
        f"{amt_param}"
        f"&tr={session_id}"
        f"&cu=INR"
    )

    return RedirectResponse(
        url=upi_intent,
        status_code=302,
    )

# -------------------------------------------------------------------------
# 8. Bank Webhook Simulation
# -------------------------------------------------------------------------

@app.post("/mock_bank_webhook")
def mock_bank_webhook(payload: dict):
    session_id = payload.get("session_id")
    status = payload.get("status")

    paid_amount = float(
        payload.get("amount", 0.0)
    )

    if (
        session_id in active_sessions
        and status == "SUCCESS"
    ):
        sess = active_sessions[session_id]

        sess["bank_webhook_received"] = 1
        sess["settled_amount"] = paid_amount
        sess["status"] = "VERIFIED"

        # Keep agent mirror synchronized after settlement.
        sync_scan(
            session_id=session_id,
            session=sess,
            ip_address="bank-webhook",
            user_agent="bank-webhook",
            spoof_probability=float(
                sess.get("classifier_confidence", 0.0)
            ),
            flagged=False,
        )

        return {
            "status": "SETTLED",
            "session_id": session_id,
            "amount": paid_amount,
        }

    return {
        "status": "REJECTED",
        "message": "Invalid transaction state",
    }

# -------------------------------------------------------------------------
# 9. Merchant Dashboard Polling
# -------------------------------------------------------------------------

@app.get("/merchant_status/{session_id}")
def merchant_status(session_id: str):
    if session_id not in active_sessions:
        return {"status": "EXPIRED"}

    sess = active_sessions[session_id]

    investigation = AGENT_STORE["investigations"].get(
        session_id
    )

    return {
        "session_id": session_id,
        "mode": sess["mode"],
        "expected_amount": sess["expected_amount"],
        "settled_amount": sess["settled_amount"],
        "status": sess["status"],
        "bank_webhook_received": sess["bank_webhook_received"],

        # New agent information for the frontend.
        "agent_status": sess.get(
            "agent_status",
            "not_triggered",
        ),
        "investigation": investigation,
    }

# -------------------------------------------------------------------------
# 10. Get Investigator Result
# -------------------------------------------------------------------------

@app.get("/agent/investigation/{session_id}")
def get_investigation(session_id: str):
    result = AGENT_STORE["investigations"].get(
        session_id
    )

    if not result:
        return {
            "status": "not_ready",
            "session_id": session_id,
        }

    return result

# -------------------------------------------------------------------------
# 11. Evidence Compiler
# -------------------------------------------------------------------------

@app.post("/dispute/{session_id}")
def create_dispute(
    session_id: str,
    background_tasks: BackgroundTasks,
    payload: dict | None = None,
):
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Unknown session_id",
        )

    reason = (
        payload.get(
            "reason",
            "Merchant disputes the payment decision",
        )
        if payload
        else "Merchant disputes the payment decision"
    )

    background_tasks.add_task(
        run_evidence_compiler,
        session_id,
        reason,
    )

    return {
        "status": "queued",
        "agent": "evidence_compiler",
        "session_id": session_id,
        "reason": reason,
    }

@app.get("/agent/evidence/{session_id}")
def get_evidence(session_id: str):
    packet = AGENT_STORE["evidence_packets"].get(
        session_id
    )

    if not packet:
        return {
            "status": "not_ready",
            "session_id": session_id,
        }

    return packet

# -------------------------------------------------------------------------
# 12. Ring Detector
# -------------------------------------------------------------------------

@app.post("/agent/ring-detector")
def trigger_ring_detector():
    result = run_ring_detector()

    return {
        "agent": "ring_detector",
        "result": result,
        "proposals": AGENT_STORE["blocklist_proposals"],
    }

# -------------------------------------------------------------------------
# 13. Merchant Copilot
# -------------------------------------------------------------------------

@app.post("/agent/copilot/{merchant_id}")
def trigger_copilot(merchant_id: str):
    result = run_merchant_copilot(
        merchant_id
    )

    return {
        "agent": "copilot",
        "merchant_id": merchant_id,
        "result": result,
        "notifications": [
            n
            for n in AGENT_STORE["notifications"]
            if n.get("merchant_id") == merchant_id
        ],
    }

# -------------------------------------------------------------------------
# 14. Red Team - SANDBOX ONLY
# -------------------------------------------------------------------------

@app.post("/agent/red-team/{sandbox_merchant_id}")
def trigger_red_team(
    sandbox_merchant_id: str,
):
    if not sandbox_merchant_id.startswith(
        "SANDBOX_"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Red-Team is sandbox-only. "
                "sandbox_merchant_id must start with SANDBOX_."
            ),
        )

    result = run_red_team_sandbox(
        sandbox_merchant_id
    )

    return {
        "agent": "red_team",
        "sandbox_merchant_id": sandbox_merchant_id,
        "result": result,
    }

# -------------------------------------------------------------------------
# 15. Agent Demo Seed
# -------------------------------------------------------------------------

@app.post("/agent/demo/seed")
def seed_agent_demo():
    seed_demo_data()

    return {
        "status": "seeded",
        "sessions": len(
            AGENT_STORE["sessions"]
        ),
        "audit_events": len(
            AGENT_STORE["audit_log"]
        ),
    }