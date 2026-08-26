import os
import io
import secrets
import time
import uuid
import joblib
import pandas as pd
import qrcode

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
merchants = {
    "MERCHANT_001": {
        "merchant_id": "MERCHANT_001",
        "business_name": "Ramesh Kirana Store",
        "merchant_name": "Ramesh Kumar",
        "mobile_number": "9876543210",
        "bank_name": "Example Bank",
        "account_holder_name": "Ramesh Kumar",
        "account_last4": "1234",
        "ifsc": "EXMP0001234",
        "verification_status": "VERIFIED",
        "created_at": time.time(),
    }
}
merchant_connections = {}
SESSION_TTL_SECONDS = 30


class MerchantRegistration(BaseModel):
    business_name: str = Field(min_length=2, max_length=120)
    merchant_name: str = Field(min_length=2, max_length=120)
    mobile_number: str = Field(pattern=r"^[0-9]{10}$")
    bank_name: str = Field(min_length=2, max_length=80)
    account_holder_name: str = Field(min_length=2, max_length=120)
    account_last4: str = Field(pattern=r"^[0-9]{4}$")
    ifsc: str = Field(min_length=4, max_length=20)


class SessionRequest(BaseModel):
    merchant_id: str
    amount: float = Field(default=50, gt=0, le=100000)


class WebhookPayload(BaseModel):
    session_id: str
    status: str
    amount: float = Field(gt=0)


class DisputePayload(BaseModel):
    reason: str = Field(default="Merchant disputes the payment decision", max_length=500)


def public_merchant(merchant):
    result = dict(merchant)
    result["mobile_number"] = f"******{merchant['mobile_number'][-4:]}"
    result["ifsc"] = f"{merchant['ifsc'][:4]}******"
    result.pop("account_holder_name", None)
    return result


def verified_merchant(merchant_id):
    merchant = merchants.get(merchant_id)
    if not merchant or merchant["verification_status"] != "VERIFIED":
        raise HTTPException(status_code=403, detail="Merchant is not verified")
    return merchant


def expire_session(session):
    if session["status"] in {"VERIFIED", "SPOOF_BLOCKED", "EXPIRED"}:
        return
    if time.time() >= session["expires_at"]:
        session["status"] = "EXPIRED"


def session_view(session, include_internal=True):
    merchant = verified_merchant(session["merchant_id"])
    result = {
        "session_id": session["session_id"],
        "merchant_id": session["merchant_id"],
        "business_name": merchant["business_name"],
        "merchant_name": merchant["merchant_name"],
        "bank_name": merchant["bank_name"],
        "account_last4": merchant["account_last4"],
        "amount": session["amount"],
        "expires_at": session["expires_at"],
        "seconds_remaining": max(0, int(session["expires_at"] - time.time())),
        "status": session["status"],
    }
    if include_internal:
        result.update({key: session.get(key) for key in (
            "risk_level", "spoof_probability", "device_fingerprint",
            "ip_request_velocity", "classifier_confidence", "verdict",
            "agent_recommendation", "agent_status")})
    return result


async def broadcast(merchant_id, event, session, **extra):
    payload = {"event": event, "session_id": session.get("session_id"),
               "merchant_id": merchant_id, "status": session.get("status"),
               "timestamp": int(time.time()), **extra}
    for connection in list(merchant_connections.get(merchant_id, set())):
        try:
            await connection.send_json(payload)
        except Exception:
            merchant_connections[merchant_id].discard(connection)

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

@app.post("/merchant/register")
def register_merchant(payload: MerchantRegistration):
    merchant_id = f"MERCHANT_{len(merchants) + 1:03d}"
    merchant = payload.model_dump()
    merchant.update({"merchant_id": merchant_id, "verification_status": "PENDING", "created_at": time.time()})
    merchants[merchant_id] = merchant
    return {"merchant": public_merchant(merchant), "message": "Complete Demo Bank Verification to continue."}


@app.post("/merchant/{merchant_id}/verify")
def verify_merchant(merchant_id: str):
    merchant = merchants.get(merchant_id)
    if not merchant:
        raise HTTPException(status_code=404, detail="Merchant not found")
    merchant["verification_status"] = "VERIFIED"
    return {"verification": "Demo Bank Verification", "status": "BANK ACCOUNT VERIFIED", "merchant": public_merchant(merchant)}


@app.get("/merchant/{merchant_id}")
def get_merchant(merchant_id: str):
    return public_merchant(verified_merchant(merchant_id))


def base_url(request):
    return os.getenv("KAVACH_BASE_URL", "").strip().rstrip("/") or f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"


@app.post("/merchant/session")
async def create_session(payload: SessionRequest, request: Request):
    merchant = verified_merchant(payload.merchant_id)
    for existing in active_sessions.values():
        expire_session(existing)
        if existing["merchant_id"] == payload.merchant_id and existing["status"] == "WAITING_FOR_SCAN":
            existing["status"] = "EXPIRED"
    session_id = f"SESS_{secrets.token_hex(4).upper()}"
    session = {
        "session_id": session_id, "merchant_id": merchant["merchant_id"],
        "created_at": time.time(), "expires_at": time.time() + SESSION_TTL_SECONDS,
        "amount": payload.amount, "expected_amount": payload.amount,
        "settled_amount": 0.0, "bank_webhook_received": 0,
        "status": "WAITING_FOR_SCAN", "ip_requests": {},
        "device_fingerprint": f"FP_{uuid.uuid4().hex[:10]}"
    }
    active_sessions[session_id] = session
    await broadcast(merchant["merchant_id"], "SESSION_CREATED", session)
    return {**session_view(session), "qr_url": f"{base_url(request)}/scan/{session_id}"}


@app.get("/merchant/session/{session_id}/qr")
def session_qr(session_id: str, request: Request):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session invalid or expired")
    expire_session(session)
    if session["status"] == "EXPIRED":
        raise HTTPException(status_code=410, detail="Session expired")
    qr = qrcode.make(f"{base_url(request)}/scan/{session_id}")
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# -------------------------------------------------------------------------
# 7. QR Scan + LightGBM Fraud Gate + Investigator Gray Zone
# -------------------------------------------------------------------------

@app.get("/scan/{session_id}")
async def handle_scan(
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
    expire_session(session)
    if session["status"] == "EXPIRED":
        await broadcast(session["merchant_id"], "SESSION_EXPIRED", session, replay_attempt=True)
        return JSONResponse(status_code=410, content={
            "error": "SESSION_EXPIRED",
            "message": "Payment session invalid or expired.",
        })
    if session["status"] != "WAITING_FOR_SCAN":
        return JSONResponse(status_code=409, content={
            "error": "SESSION_ALREADY_USED",
            "status": session["status"],
        })

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
    session["spoof_probability"] = spoof_prob
    session["risk_level"] = "HIGH" if is_spoof else "LOW"
    session["verdict"] = "BLOCKED" if is_spoof else "ALLOW"
    await broadcast(session["merchant_id"], "SCAN_RECEIVED", session)
    await broadcast(session["merchant_id"], "THREAT_ANALYSIS", session)

    # -------------------------------------------------------------
    # Synchronize this real FastAPI session with agent STORE.
    # -------------------------------------------------------------

    sync_scan(
        session_id=session_id,
        session_data=session,
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
        await broadcast(session["merchant_id"], "SPOOF_BLOCKED", session)

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

    session["status"] = "PAYMENT_PENDING"
    await broadcast(session["merchant_id"], "PAYMENT_PENDING", session)
    merchant = verified_merchant(session["merchant_id"])

    amt_param = (
        f"&am={session['expected_amount']}"
        if session["expected_amount"] > 0
        else ""
    )

    upi_intent = (
        f"upi://pay?"
        f"pa={merchant['merchant_id']}@okaxis"
        f"&pn={merchant['business_name']}"
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
async def mock_bank_webhook(payload: WebhookPayload):
    if payload.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = active_sessions[payload.session_id]
    if sess["status"] != "PAYMENT_PENDING" or payload.status != "SUCCESS" or payload.amount != sess["amount"]:
        raise HTTPException(status_code=409, detail="Invalid transaction state or amount")

    sess["bank_webhook_received"] = 1
    sess["settled_amount"] = payload.amount
    sess["status"] = "VERIFIED"
    sess["verdict"] = "VERIFIED"

    sync_scan(
        session_id=payload.session_id,
        session_data=sess,
        ip_address="bank-webhook",
        user_agent="bank-webhook",
        spoof_probability=float(sess.get("classifier_confidence", 0.0)),
        flagged=False,
    )
    await broadcast(sess["merchant_id"], "PAYMENT_SETTLED", sess)
    return {"status": "SETTLED", "session_id": payload.session_id, "amount": payload.amount}

# -------------------------------------------------------------------------
# 9. Customer-safe session lookup and merchant dashboard polling
# -------------------------------------------------------------------------

@app.get("/customer/session/{session_id}")
def customer_session(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Payment session invalid")
    expire_session(session)
    return session_view(session, include_internal=False)


@app.get("/customer/session/{session_id}/next")
def next_customer_session(session_id: str):
    current = active_sessions.get(session_id)
    if not current:
        raise HTTPException(status_code=404, detail="Payment session invalid")
    for candidate in reversed(list(active_sessions.values())):
        expire_session(candidate)
        if candidate["merchant_id"] == current["merchant_id"] and candidate["status"] == "WAITING_FOR_SCAN":
            return session_view(candidate, include_internal=False)
    raise HTTPException(status_code=404, detail="No replacement session is available yet")

@app.get("/merchant_status/{session_id}")
def merchant_status(session_id: str):
    if session_id not in active_sessions:
        return {"status": "EXPIRED"}

    sess = active_sessions[session_id]
    expire_session(sess)
    result = session_view(sess)
    result.update({
        "settled_amount": sess.get("settled_amount", 0.0),
        "bank_webhook_received": sess.get("bank_webhook_received", 0),
        "investigation": AGENT_STORE["investigations"].get(session_id),
    })
    return result

# -------------------------------------------------------------------------
# 10. Get Investigator Result
# -------------------------------------------------------------------------

@app.post("/agent/investigation/{session_id}")
def trigger_investigation(session_id: str, background_tasks: BackgroundTasks):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    probability = float(session.get("spoof_probability", 0.5))
    background_tasks.add_task(run_investigator_gray_zone, session_id, probability)
    return {"status": "queued", "agent": "investigator", "session_id": session_id}

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
    payload: DisputePayload,
):
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Unknown session_id",
        )

    background_tasks.add_task(
        run_evidence_compiler,
        session_id,
        payload.reason,
    )

    return {
        "status": "queued",
        "agent": "evidence_compiler",
        "session_id": session_id,
        "reason": payload.reason,
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
    verified_merchant(merchant_id)
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


@app.websocket("/ws/merchant/{merchant_id}")
async def merchant_websocket(websocket: WebSocket, merchant_id: str):
    if merchant_id not in merchants:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    merchant_connections.setdefault(merchant_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        merchant_connections[merchant_id].discard(websocket)