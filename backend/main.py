import os
import io
import time
import uuid
import asyncio
import socket
from pathlib import Path

import joblib
import pandas as pd
import qrcode

from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.responses import (
    StreamingResponse,
    RedirectResponse,
    JSONResponse,
    HTMLResponse,
    FileResponse,
)

from fastapi.middleware.cors import CORSMiddleware


# =========================================================================
# CONFIGURATION
# =========================================================================

# IMPORTANT:
# For phone testing on the same Wi-Fi, set:
#
# Windows PowerShell:
# $env:KAVACH_BASE_URL="http://192.168.1.15:8000"
#
# Or replace the IP with your computer's actual IPv4 address.
#
# For ngrok:
# $env:KAVACH_BASE_URL="https://your-real-ngrok-url.ngrok-free.app"

KAVACH_BASE_URL = os.getenv(
    "KAVACH_BASE_URL",
    "",
)

SESSION_TTL_SECONDS = 30


# =========================================================================
# BASE URL RESOLUTION
# =========================================================================

def resolve_public_base_url(request: Request | None = None) -> str:
    """
    Resolve the URL that will be embedded inside the QR code.

    Priority:
        1. KAVACH_BASE_URL environment variable
        2. Incoming request Host header
        3. localhost fallback

    For phone testing, KAVACH_BASE_URL should normally be:

        http://192.168.1.15:8000

    NOT:

        http://127.0.0.1:8000
        http://localhost:8000
    """

    if KAVACH_BASE_URL:
        return KAVACH_BASE_URL.rstrip("/")

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        lan_ip = probe.getsockname()[0]
        probe.close()
        if not lan_ip.startswith("127."):
            return f"http://{lan_ip}:8000"
    except OSError:
        pass

    if request is not None:
        host = request.headers.get("host")

        if host:
            return f"http://{host}"

    return "http://localhost:8000"


# =========================================================================
# MERCHANT WEBSOCKET HUB
# =========================================================================

class MerchantSocketHub:
    """
    Maintains WebSocket connections for merchant dashboards.

    Each merchant can have one or more dashboard connections.

    Example:

        MERCHANT_001
            |
            +--- Dashboard Browser
            +--- Another Dashboard Browser
    """

    def __init__(self):
        self.connections: dict[str, set[WebSocket]] = {}
        self.loop = None

    async def connect(
        self,
        merchant_id: str,
        websocket: WebSocket,
    ):
        await websocket.accept()

        self.loop = asyncio.get_running_loop()

        self.connections.setdefault(
            merchant_id,
            set(),
        ).add(websocket)

    def disconnect(
        self,
        merchant_id: str,
        websocket: WebSocket,
    ):
        sockets = self.connections.get(
            merchant_id
        )

        if not sockets:
            return

        sockets.discard(websocket)

        if not sockets:
            self.connections.pop(
                merchant_id,
                None,
            )

    async def broadcast(
        self,
        merchant_id: str,
        event: dict,
    ):
        sockets = tuple(
            self.connections.get(
                merchant_id,
                (),
            )
        )

        stale = []

        for websocket in sockets:
            try:
                await websocket.send_json(event)

            except Exception:
                stale.append(websocket)

        for websocket in stale:
            self.disconnect(
                merchant_id,
                websocket,
            )


socket_hub = MerchantSocketHub()


def schedule_event(
    merchant_id: str,
    event: dict,
):
    """
    Send an event to the merchant dashboard.

    This allows backend events such as:

        SESSION_CREATED
        SCAN_RECEIVED
        THREAT_ANALYSIS
        SPOOF_BLOCKED
        PAYMENT_SETTLED
        SESSION_EXPIRED

    to appear immediately on the merchant dashboard.
    """

    if (
        socket_hub.loop
        and socket_hub.loop.is_running()
    ):
        asyncio.run_coroutine_threadsafe(
            socket_hub.broadcast(
                merchant_id,
                event,
            ),
            socket_hub.loop,
        )

        return

    try:
        loop = asyncio.get_running_loop()

    except RuntimeError:
        return

    loop.create_task(
        socket_hub.broadcast(
            merchant_id,
            event,
        )
    )


# =========================================================================
# AGENT IMPORTS
# =========================================================================

# Agents are advisory/defense-only.
# They do not directly control real financial transactions.

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


# =========================================================================
# FASTAPI APPLICATION
# =========================================================================

app = FastAPI(
    title="KavachQR Dual-Mode Fraud Defense Gateway",
    version="2.0.0",
    description=(
        "Prototype QR payment security gateway "
        "with ML fraud detection and multi-agent defense."
    ),
)


# =========================================================================
# CORS
# =========================================================================

app.add_middleware(
    CORSMiddleware,

    # Local development origins.
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://192.168.1.15:5500",
    ],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],

    expose_headers=[
        "X-Session-ID",
        "X-Gateway-URL",
        "X-Merchant-ID",
    ],
)


# =========================================================================
# ML MODEL
# =========================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lgbm_fraud_model.pkl",
)


CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_config.pkl",
)


if (
    os.path.exists(MODEL_PATH)
    and os.path.exists(CONFIG_PATH)
):

    model = joblib.load(
        MODEL_PATH
    )

    config = joblib.load(
        CONFIG_PATH
    )

    OPTIMAL_THRESHOLD = config.get(
        "threshold",
        0.5,
    )

    print(
        "ML model loaded successfully."
    )

    print(
        f"Optimal threshold: "
        f"{OPTIMAL_THRESHOLD:.2f}"
    )

else:

    model = None

    OPTIMAL_THRESHOLD = 0.5

    print(
        "WARNING: ML model artifacts "
        "not found."
    )

    print(
        "Running with fallback detection rules."
    )


# =========================================================================
# SESSION STORAGE
# =========================================================================

active_sessions: dict[str, dict] = {}


# =========================================================================
# SESSION EXPIRATION
# =========================================================================

def expire_sessions(
    now: float | None = None,
):
    """
    Expire sessions that passed the 30-second
    security window.
    """

    current_time = (
        now
        if now is not None
        else time.time()
    )

    for session in list(
        active_sessions.values()
    ):

        if (
            session["status"]
            not in {
                "VERIFIED",
                "SPOOF_BLOCKED",
                "EXPIRED",
            }
            and current_time
            >= session["expires_at"]
        ):

            session["status"] = "EXPIRED"

            schedule_event(
                session["merchant_id"],
                {
                    "event": "SESSION_EXPIRED",

                    "session_id":
                        session["session_id"],

                    "status": "EXPIRED",
                },
            )


# =========================================================================
# CREATE SESSION
# =========================================================================

def create_session(
    merchant_id: str,
    amount: float,
):
    """
    Create one active secure QR session.

    A merchant cannot accidentally create
    multiple simultaneous active sessions.
    """

    expire_sessions()

    # Check whether this merchant already
    # has an active session.

    for session in active_sessions.values():

        if (
            session["merchant_id"]
            == merchant_id

            and session["status"]
            not in {
                "EXPIRED",
                "VERIFIED",
                "SPOOF_BLOCKED",
            }
        ):

            return session, False


    created_at = time.time()

    session_id = (
        f"SESS_"
        f"{uuid.uuid4().hex[:8].upper()}"
    )


    session = {

        "session_id":
            session_id,

        "created_at":
            created_at,

        "expires_at":
            created_at
            + SESSION_TTL_SECONDS,

        "merchant_id":
            merchant_id,

        "mode":
            (
                "FIXED"
                if amount > 0
                else "ZERO_INPUT"
            ),

        "expected_amount":
            amount,

        # Kept for compatibility
        # with existing agent code.
        "amount":
            amount,

        "settled_amount":
            0.0,

        "bank_webhook_received":
            0,

        "status":
            "WAITING_FOR_SCAN",

        "ip_requests":
            {},

        "agent_status":
            "not_triggered",

        "investigator_status":
            None,

        "classifier_confidence":
            None,

        "scan_time_delta":
            None,

        "ip_request_velocity":
            None,

        "user_agent_score":
            None,
    }


    active_sessions[
        session_id
    ] = session


    schedule_event(
        merchant_id,
        {
            "event":
                "SESSION_CREATED",

            "session_id":
                session_id,

            "status":
                session["status"],

            "expires_at":
                session["expires_at"],

            "amount":
                amount,
        },
    )


    print(
        f"> Generated "
        f"[Zero-Touch Dynamic] "
        f"Session: {session_id}"
    )


    return session, True


# =========================================================================
# ROOT / HEALTH CHECK
# =========================================================================

@app.get("/")
def root():

    return {

        "service":
            "KavachQR Dual-Mode "
            "Fraud Defense Gateway",

        "status":
            "online",

        "ml_model_loaded":
            model is not None,

        "threshold":
            OPTIMAL_THRESHOLD,

        "session_ttl_seconds":
            SESSION_TTL_SECONDS,

        "qr_base_url":
            (
                KAVACH_BASE_URL
                if KAVACH_BASE_URL
                else "AUTO"
            ),

        "agents": [

            "investigator",

            "evidence_compiler",

            "ring_detector",

            "copilot",

            "red_team",

        ],

        "websocket":
            "/ws/merchant/{merchant_id}",

        "agent_policy":
            "advisory / defense-only",
    }


# =========================================================================
# GENERATE QR
# =========================================================================

@app.get("/generate_qr")
def generate_qr(
    request: Request,
    merchant_id: str = "MERCHANT_001",
    amount: float = 50.0,
):
    """
    Generate a secure dynamic QR.

    The QR contains:

        http://192.168.1.15:8000/scan/SESS_XXXXXXXX

    The session is created through create_session(),
    ensuring that every generated session has the
    complete structure needed by the rest of the backend.
    """

    # ---------------------------------------------------------
    # Create/reuse session
    # ---------------------------------------------------------

    session, created = create_session(
        merchant_id=merchant_id,
        amount=amount,
    )

    session_id = session[
        "session_id"
    ]


    # ---------------------------------------------------------
    # Resolve phone-accessible URL
    # ---------------------------------------------------------

    public_base = (
        resolve_public_base_url(
            request
        )
    )


    gateway_url = (
        f"{public_base}"
        f"/scan/{session_id}"
    )


    print()
    print("=" * 70)
    print("KAVACHQR - SECURE QR GENERATED")
    print("=" * 70)
    print(
        f"Merchant       : {merchant_id}"
    )
    print(
        f"Amount         : ₹{amount:.2f}"
    )
    print(
        f"Session        : {session_id}"
    )
    print(
        f"New Session    : {created}"
    )
    print(
        f"Expires At     : {session['expires_at']}"
    )
    print(
        f"Gateway URL    : {gateway_url}"
    )
    print("=" * 70)
    print()


    # ---------------------------------------------------------
    # Generate QR
    # ---------------------------------------------------------

    qr = qrcode.QRCode(
        version=1,
        error_correction=
            qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )


    qr.add_data(
        gateway_url
    )

    qr.make(
        fit=True
    )


    img = qr.make_image(
        fill_color="black",
        back_color="white",
    )


    # ---------------------------------------------------------
    # Convert image to memory
    # ---------------------------------------------------------

    buf = io.BytesIO()

    img.save(
        buf,
        format="PNG",
    )

    buf.seek(0)


    # ---------------------------------------------------------
    # Return QR image
    # ---------------------------------------------------------

    return StreamingResponse(

        buf,

        media_type="image/png",

        headers={

            "X-Session-ID":
                session_id,

            "X-Gateway-URL":
                gateway_url,

            "X-Merchant-ID":
                merchant_id,
        },
    )


# =========================================================================
# SESSION QR INFORMATION
# =========================================================================

@app.get("/qr/{session_id}")
def get_session_qr(
    session_id: str,
    request: Request,
):

    expire_sessions()


    if session_id not in active_sessions:

        raise HTTPException(
            status_code=404,
            detail="Unknown session_id",
        )


    session = active_sessions[
        session_id
    ]


    public_base = (
        resolve_public_base_url(
            request
        )
    )


    return {

        "session_id":
            session_id,

        "qr_url":
            (
                f"{public_base}"
                f"/scan/{session_id}"
            ),

        "merchant_id":
            session["merchant_id"],

        "mode":
            session["mode"],

        "expected_amount":
            session["expected_amount"],

        "status":
            session["status"],

        "expires_in":
            max(
                0,
                round(
                    session["expires_at"]
                    - time.time()
                ),
            ),

        "expires_at":
            session["expires_at"],
    }


# =========================================================================
# MERCHANT SESSION
# =========================================================================

@app.get("/merchant_session/{merchant_id}")
def merchant_session(
    merchant_id: str,
    request: Request,
    amount: float = 50.0,
):

    session, created = (
        create_session(
            merchant_id,
            amount,
        )
    )


    public_base = (
        resolve_public_base_url(
            request
        )
    )


    return {

        "session_id":
            session["session_id"],

        "status":
            session["status"],

        "amount":
            session["expected_amount"],

        "expires_at":
            session["expires_at"],

        "expires_in":
            max(
                0,
                round(
                    session["expires_at"]
                    - time.time()
                ),
            ),

        "created":
            created,

        "qr_url":
            (
                f"{public_base}"
                f"/scan/"
                f"{session['session_id']}"
            ),
    }


# =========================================================================
# CUSTOMER DISPLAY
# =========================================================================

@app.get("/customer-display")
def customer_display():

    page = Path(
        BASE_DIR,
        "frontend",
        "customer-display.html",
    )


    if not page.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Customer display "
                "is unavailable"
            ),
        )


    return FileResponse(
        page
    )


# =========================================================================
# MOBILE PAYMENT PAGE
# =========================================================================

@app.get(
    "/mobile-pay/{session_id}"
)
def mobile_pay(
    session_id: str,
):

    expire_sessions()


    if (
        session_id
        not in active_sessions
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Payment session "
                "expired"
            ),
        )


    page = Path(
        BASE_DIR,
        "templates",
        "mobile_pay.html",
    )


    if not page.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Payment page "
                "is unavailable"
            ),
        )


    return FileResponse(
        page
    )


# =========================================================================
# QR SCAN
# =========================================================================

def analyze_scan(
    session: dict,
    request_signals: dict,
) -> dict:
    """Run the authoritative LightGBM or fallback scan decision."""
    features = pd.DataFrame([
        {
            "scan_time_delta": request_signals["scan_time_delta"],
            "ip_request_velocity": request_signals["ip_request_velocity"],
            "user_agent_score": request_signals["user_agent_score"],
            "bank_webhook_received": session["bank_webhook_received"],
        }
    ])

    if model is not None:
        try:
            spoof_probability = float(model.predict_proba(features)[0][1])
            is_spoof = spoof_probability >= OPTIMAL_THRESHOLD
        except Exception as exc:
            print("ML prediction failed:", exc)
            is_spoof = (
                request_signals["scan_time_delta"] > 60
                or request_signals["ip_request_velocity"] > 5
                or request_signals["user_agent_score"] < 0.5
            )
            spoof_probability = 0.90 if is_spoof else 0.10
    else:
        is_spoof = (
            request_signals["scan_time_delta"] > 60
            or request_signals["ip_request_velocity"] > 5
            or request_signals["user_agent_score"] < 0.5
        )
        spoof_probability = 0.90 if is_spoof else 0.10

    session.update(request_signals)
    session["optimal_threshold"] = OPTIMAL_THRESHOLD
    session["classifier_confidence"] = spoof_probability
    session["flagged"] = is_spoof
    return {
        "spoof_probability": spoof_probability,
        "is_spoof": is_spoof,
    }

@app.get(
    "/scan/{session_id}"
)
def handle_scan(
    session_id: str,
    request: Request,
    background_tasks:
        BackgroundTasks,
):

    expire_sessions()


    # ---------------------------------------------------------
    # Validate session
    # ---------------------------------------------------------

    if (
        session_id
        not in active_sessions
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Session Expired "
                "or Invalid"
            ),
        )


    session = active_sessions[
        session_id
    ]


    # ---------------------------------------------------------
    # Prevent replay / duplicate usage
    # ---------------------------------------------------------

    if session["status"] in {

        "EXPIRED",

        "VERIFIED",

        "SPOOF_BLOCKED",

    }:

        return HTMLResponse(

            """
            <html>
            <body style="
                font-family:Arial;
                text-align:center;
                padding:50px;
            ">

            <h1>
                🔒 Payment Session Unavailable
            </h1>

            <p>
                This KavachQR security session
                has expired or has already been used.
            </p>

            </body>
            </html>
            """,

            status_code=410,
        )


    # ---------------------------------------------------------
    # Calculate scan timing
    # ---------------------------------------------------------

    scan_time_delta = (
        time.time()
        - session["created_at"]
    )


    # ---------------------------------------------------------
    # Client information
    # ---------------------------------------------------------

    client_ip = (

        request.client.host

        if request.client

        else "unknown"
    )


    session["ip_requests"][client_ip] = (

        session["ip_requests"].get(
            client_ip,
            0,
        )
        + 1
    )


    ip_velocity = (
        session["ip_requests"][
            client_ip
        ]
    )


    user_agent = (
        request.headers.get(
            "user-agent",
            "",
        )
        .lower()
    )


    # ---------------------------------------------------------
    # User-agent signal
    # ---------------------------------------------------------

    user_agent_score = (

        0.30

        if (
            "python" in user_agent
            or "curl" in user_agent
            or "requests" in user_agent
        )

        else 0.95
    )


    session[
        "status"
    ] = "ANALYZING"


    # ---------------------------------------------------------
    # WebSocket event
    # ---------------------------------------------------------

    schedule_event(

        session["merchant_id"],

        {

            "event":
                "SCAN_RECEIVED",

            "session_id":
                session_id,

            "status":
                "ANALYZING",

            "scan_time_delta":
                round(
                    scan_time_delta,
                    3,
                ),
        },
    )


    analysis = analyze_scan(
        session,
        {
            "scan_time_delta": scan_time_delta,
            "ip_request_velocity": ip_velocity,
            "user_agent_score": user_agent_score,
        },
    )
    spoof_prob = analysis["spoof_probability"]
    is_spoof = analysis["is_spoof"]


    # =========================================================
    # SYNCHRONIZE WITH AGENTS
    # =========================================================

    try:

        sync_scan(

            session_id=session_id,

            session_data=session,
        )

    except Exception as e:

        print(
            "Agent sync warning:",
            e,
        )


    # =========================================================
    # INVESTIGATOR GRAY ZONE
    # =========================================================

    if (
        0.35
        <= spoof_prob
        <= 0.65
    ):

        session[
            "agent_status"
        ] = "investigator_queued"


        background_tasks.add_task(

            run_investigator_gray_zone,

            session_id,

            spoof_prob,
        )

    else:

        session[
            "agent_status"
        ] = "not_triggered"


    # =========================================================
    # SPOOF BLOCK
    # =========================================================

    if is_spoof:

        session[
            "status"
        ] = "SPOOF_BLOCKED"


        schedule_event(

            session["merchant_id"],

            {

                "event":
                    "SPOOF_BLOCKED",

                "session_id":
                    session_id,

                "status":
                    "SPOOF_BLOCKED",

                "spoof_probability":
                    round(
                        spoof_prob,
                        4,
                    ),

                "risk_level":
                    "HIGH",

                "verdict":
                    "SPOOF",

                "scan_time_delta":
                    round(
                        scan_time_delta,
                        3,
                    ),

                "ip_request_velocity":
                    ip_velocity,

                "user_agent_score":
                    user_agent_score,
            },
        )


        return JSONResponse(

            status_code=403,

            content={

                "error":
                    "SECURITY_ALERT_SPOOF_BLOCKED",

                "message":
                    (
                        "Offline spoof app "
                        "detected by "
                        "KavachQR Engine."
                    ),

                "session_id":
                    session_id,

                "spoof_probability":
                    round(
                        spoof_prob,
                        4,
                    ),

                "agent_status":
                    session[
                        "agent_status"
                    ],
            },
        )


    # =========================================================
    # GENUINE PAYMENT
    # =========================================================

    session[
        "status"
    ] = "PAYMENT_PENDING"


    schedule_event(

        session["merchant_id"],

        {

            "event":
                "THREAT_ANALYSIS",

            "session_id":
                session_id,

            "status":
                "PAYMENT_PENDING",

            "spoof_probability":
                round(
                    spoof_prob,
                    4,
                ),

            "risk_level":
                "LOW",

            "verdict":
                "GENUINE",

            "scan_time_delta":
                round(
                    scan_time_delta,
                    3,
                ),

            "ip_request_velocity":
                ip_velocity,

            "user_agent_score":
                user_agent_score,

            "bank_webhook_received":
                session[
                    "bank_webhook_received"
                ],
        },
    )


    # ---------------------------------------------------------
    # Build UPI intent
    # ---------------------------------------------------------

    merchant_id = (
        session["merchant_id"]
    )


    if (
        session["expected_amount"]
        > 0
    ):

        amt_param = (
            f"&am="
            f"{session['expected_amount']}"
        )

    else:

        amt_param = ""


    upi_intent = (

        "upi://pay?"

        f"pa={merchant_id}@okaxis"

        "&pn=KiranaStore"

        f"{amt_param}"

        f"&tr={session_id}"

        "&cu=INR"
    )


    return RedirectResponse(

        url=upi_intent,

        status_code=302,
    )


@app.post("/agent/demo/spoof-test/{session_id}")
def spoof_test(
    session_id: str,
):
    """Run a clearly labeled synthetic scan through the real classifier."""
    expire_sessions()
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if session["status"] in {"EXPIRED", "VERIFIED", "SPOOF_BLOCKED"}:
        raise HTTPException(status_code=410, detail="Session is no longer testable")

    session["status"] = "ANALYZING"
    synthetic_signals = {
        "scan_time_delta": SESSION_TTL_SECONDS + 1,
        "ip_request_velocity": session["ip_requests"].get("sandbox", 0) + 6,
        "user_agent_score": 0.30,
        "sandbox_simulation": True,
    }
    analysis = analyze_scan(session, synthetic_signals)
    spoof_probability = analysis["spoof_probability"]
    is_spoof = analysis["is_spoof"]
    session["status"] = "SPOOF_BLOCKED" if is_spoof else "PAYMENT_PENDING"

    sync_scan(session_id=session_id, session_data=session)
    event = {
        "event": "SPOOF_BLOCKED" if is_spoof else "THREAT_ANALYSIS",
        "session_id": session_id,
        "status": session["status"],
        "spoof_probability": round(spoof_probability, 4),
        "risk_level": "HIGH" if is_spoof else "LOW",
        "verdict": "SPOOF DETECTED" if is_spoof else "GENUINE",
        "sandbox_simulation": True,
        **synthetic_signals,
    }
    schedule_event(session["merchant_id"], event)
    return {
        "simulation": "SANDBOX",
        "session_id": session_id,
        "status": session["status"],
        "spoof_probability": round(spoof_probability, 4),
        "risk_level": event["risk_level"],
        "verdict": event["verdict"],
        "signals": synthetic_signals,
    }


# =========================================================================
# MOCK BANK WEBHOOK
# =========================================================================

@app.post(
    "/mock_bank_webhook"
)
def mock_bank_webhook(
    payload: dict,
):

    expire_sessions()


    session_id = payload.get(
        "session_id"
    )


    status = payload.get(
        "status"
    )


    try:

        paid_amount = float(
            payload.get(
                "amount",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return {

            "status":
                "REJECTED",

            "message":
                "Invalid amount",
        }


    # ---------------------------------------------------------
    # Validate session
    # ---------------------------------------------------------

    if (
        session_id
        not in active_sessions
    ):

        return {

            "status":
                "REJECTED",

            "message":
                "Unknown session",
        }


    sess = active_sessions[
        session_id
    ]


    # ---------------------------------------------------------
    # Validate status
    # ---------------------------------------------------------

    if status != "SUCCESS":

        return {

            "status":
                "REJECTED",

            "message":
                "Bank transaction unsuccessful",
        }


    # ---------------------------------------------------------
    # Validate payment state
    # ---------------------------------------------------------

    if (
        sess["status"]
        != "PAYMENT_PENDING"
    ):

        return {

            "status":
                "REJECTED",

            "message":
                (
                    "Session is not "
                    "awaiting payment"
                ),
        }


    # ---------------------------------------------------------
    # Validate amount
    # ---------------------------------------------------------

    expected_amount = (
        sess["expected_amount"]
    )


    if (
        expected_amount > 0

        and abs(
            paid_amount
            - expected_amount
        ) > 0.01
    ):

        return {

            "status":
                "REJECTED",

            "message":
                "Amount mismatch",

            "expected_amount":
                expected_amount,

            "received_amount":
                paid_amount,
        }


    # ---------------------------------------------------------
    # Settlement
    # ---------------------------------------------------------

    sess[
        "bank_webhook_received"
    ] = 1


    sess[
        "settled_amount"
    ] = paid_amount


    sess[
        "status"
    ] = "VERIFIED"


    # ---------------------------------------------------------
    # WebSocket settlement event
    # ---------------------------------------------------------

    schedule_event(

        sess["merchant_id"],

        {

            "event":
                "PAYMENT_SETTLED",

            "session_id":
                session_id,

            "status":
                "VERIFIED",

            "settled_amount":
                paid_amount,

            "bank_webhook_received":
                1,
        },
    )


    # ---------------------------------------------------------
    # Synchronize agents
    # ---------------------------------------------------------

    try:

        sync_scan(

            session_id=session_id,

            session_data=sess,
        )

    except Exception as e:

        print(
            "Agent settlement sync warning:",
            e,
        )


    return {

        "status":
            "SETTLED",

        "session_id":
            session_id,

        "amount":
            paid_amount,
    }


# =========================================================================
# MERCHANT STATUS
# =========================================================================

@app.get(
    "/merchant_status/{session_id}"
)
def merchant_status(
    session_id: str,
):

    expire_sessions()


    if (
        session_id
        not in active_sessions
    ):

        return {
            "status":
                "EXPIRED"
        }


    sess = active_sessions[
        session_id
    ]


    investigation = (
        AGENT_STORE[
            "investigations"
        ].get(
            session_id
        )
    )


    return {

        "session_id":
            session_id,

        "mode":
            sess["mode"],

        "expected_amount":
            sess["expected_amount"],

        "settled_amount":
            sess["settled_amount"],

        "status":
            sess["status"],

        "bank_webhook_received":
            sess[
                "bank_webhook_received"
            ],

        "expires_at":
            sess["expires_at"],

        "expires_in":
            max(
                0,
                round(
                    sess["expires_at"]
                    - time.time()
                ),
            ),

        "scan_time_delta":
            sess.get(
                "scan_time_delta"
            ),

        "ip_request_velocity":
            sess.get(
                "ip_request_velocity"
            ),

        "user_agent_score":
            sess.get(
                "user_agent_score"
            ),

        "spoof_probability":
            sess.get(
                "classifier_confidence"
            ),

        "merchant_id":
            sess["merchant_id"],

        "agent_status":
            sess.get(
                "agent_status",
                "not_triggered",
            ),

        "investigation":
            investigation,
    }


# =========================================================================
# MERCHANT WEBSOCKET
# =========================================================================

@app.websocket(
    "/ws/merchant/{merchant_id}"
)
async def merchant_websocket(
    websocket: WebSocket,
    merchant_id: str,
):

    await socket_hub.connect(
        merchant_id,
        websocket,
    )


    try:

        await websocket.send_json(
            {
                "event":
                    "CONNECTED",

                "merchant_id":
                    merchant_id,
            }
        )


        while True:

            # Keep connection alive.
            await websocket.receive_text()


    except WebSocketDisconnect:

        socket_hub.disconnect(
            merchant_id,
            websocket,
        )

    except Exception:

        socket_hub.disconnect(
            merchant_id,
            websocket,
        )


# =========================================================================
# INVESTIGATOR
# =========================================================================

@app.get(
    "/agent/investigation/{session_id}"
)
def get_investigation(
    session_id: str,
):

    result = (
        AGENT_STORE[
            "investigations"
        ].get(
            session_id
        )
    )


    if not result:

        return {

            "status":
                "not_ready",

            "session_id":
                session_id,
        }


    return result


@app.post(
    "/agent/investigation/{session_id}"
)
def run_investigation(
    session_id: str,
    background_tasks:
        BackgroundTasks,
):

    if (
        session_id
        not in active_sessions
    ):

        raise HTTPException(
            status_code=404,
            detail="Unknown session_id",
        )


    session = active_sessions[
        session_id
    ]


    session[
        "agent_status"
    ] = "investigator_queued"


    background_tasks.add_task(

        run_investigator_gray_zone,

        session_id,

        session.get(
            "classifier_confidence",
            0.0,
        ),
    )


    return {

        "status":
            "queued",

        "agent":
            "investigator",

        "session_id":
            session_id,
    }


# =========================================================================
# EVIDENCE COMPILER
# =========================================================================

@app.post(
    "/dispute/{session_id}"
)
def create_dispute(
    session_id: str,

    background_tasks:
        BackgroundTasks,

    payload: dict | None = None,
):

    if (
        session_id
        not in active_sessions
    ):

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

        else
        "Merchant disputes the payment decision"
    )


    background_tasks.add_task(

        run_evidence_compiler,

        session_id,
    )


    return {

        "status":
            "queued",

        "agent":
            "evidence_compiler",

        "session_id":
            session_id,

        "reason":
            reason,
    }


@app.get(
    "/agent/evidence/{session_id}"
)
def get_evidence(
    session_id: str,
):

    packet = (
        AGENT_STORE[
            "evidence_packets"
        ].get(
            session_id
        )
    )


    if not packet:

        return {

            "status":
                "not_ready",

            "session_id":
                session_id,
        }


    return packet


# =========================================================================
# RING DETECTOR
# =========================================================================

@app.post(
    "/agent/ring-detector"
)
def trigger_ring_detector():

    result = (
        run_ring_detector()
    )


    return {

        "agent":
            "ring_detector",

        "result":
            result,

        "proposals":
            AGENT_STORE[
                "blocklist_proposals"
            ],
    }


# =========================================================================
# MERCHANT COPILOT
# =========================================================================

@app.post(
    "/agent/copilot/{merchant_id}"
)
def trigger_copilot(
    merchant_id: str,
):

    result = (
        run_merchant_copilot(
            merchant_id
        )
    )


    return {

        "agent":
            "copilot",

        "merchant_id":
            merchant_id,

        "result":
            result,

        "notifications": [

            n

            for n in
            AGENT_STORE[
                "notifications"
            ]

            if (
                n.get(
                    "merchant_id"
                )
                == merchant_id
            )
        ],
    }


# =========================================================================
# RED TEAM SANDBOX
# =========================================================================

@app.post(
    "/agent/red-team/{sandbox_merchant_id}"
)
def trigger_red_team(
    sandbox_merchant_id: str,
):

    # Red Team is intentionally
    # restricted to sandbox merchants.

    if not sandbox_merchant_id.startswith(
        "SANDBOX_"
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Red-Team is sandbox-only. "
                "sandbox_merchant_id must "
                "start with SANDBOX_."
            ),
        )


    result = (
        run_red_team_sandbox(
            sandbox_merchant_id
        )
    )


    return {

        "agent":
            "red_team",

        "sandbox_merchant_id":
            sandbox_merchant_id,

        "result":
            result,
    }


# =========================================================================
# AGENT DEMO SEED
# =========================================================================

@app.post(
    "/agent/demo/seed"
)
def seed_agent_demo():

    seed_demo_data()


    return {

        "status":
            "seeded",

        "sessions":
            len(
                AGENT_STORE[
                    "sessions"
                ]
            ),

        "audit_events":
            len(
                AGENT_STORE[
                    "audit_log"
                ]
            ),
    }


# =========================================================================
# DEBUG ENDPOINT - DEVELOPMENT ONLY
# =========================================================================

@app.get("/debug/sessions")
def debug_sessions():

    """
    Development endpoint.

    Shows active sessions and their current states.

    Remove this endpoint before deploying publicly.
    """

    expire_sessions()

    return {

        "count":
            len(active_sessions),

        "sessions":
            list(
                active_sessions.values()
            ),
    }