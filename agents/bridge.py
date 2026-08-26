"""
KavachQR Agent Bridge

Connects the FastAPI payment gateway with the defense-only
multi-agent system.

IMPORTANT:
- Agents are advisory.
- Agents do NOT directly block payments.
- Final payment decisions remain with the deterministic
  ML/security gateway.
"""

from typing import Any, Dict, Optional

from agents.tool_implementations import (
    STORE,
    sync_fastapi_session,
    get_session_details,
    get_flagged_sessions_batch,
    cluster_by_fingerprint_similarity,
    propose_blocklist_entry,
    submit_investigation_verdict,
)


# ============================================================
# Helper
# ============================================================

def _safe_call(function, *args, **kwargs):
    """
    Execute an agent tool safely.

    Agent failures should never bring down the payment gateway.
    """
    try:
        return function(*args, **kwargs)

    except Exception as exc:
        return {
            "status": "agent_error",
            "error": str(exc),
            "tool": getattr(function, "__name__", "unknown"),
        }


# ============================================================
# 1. Synchronize FastAPI session -> Agent Store
# ============================================================

def sync_scan(
    session_id: str,
    session_data: Dict[str, Any],
    **metadata: Any,
) -> Dict[str, Any]:
    """
    Copies the current FastAPI session into the agent store.

    This keeps the agent layer separate from the payment gateway.
    """

    session_data = dict(session_data)
    session_data.update(metadata)

    result = _safe_call(
        sync_fastapi_session,
        session_id=session_id,
        session_data=session_data,
    )

    if isinstance(result, dict) and result.get("status") == "agent_error":
        return {
            "status": "sync_error",
            "session_id": session_id,
            "error": result.get("error"),
        }

    return {
        "status": "synced",
        "session_id": session_id,
        "result": result,
    }


# ============================================================
# 2. Investigator Agent
# ============================================================

def run_investigator_gray_zone(
    session_id: str,
    spoof_probability: float,
) -> Dict[str, Any]:
    """
    Investigates a gray-zone transaction.

    The Investigator is advisory only.
    It does NOT block the payment.
    """

    session = _safe_call(
        get_session_details,
        session_id,
    )

    if session.get("status") == "agent_error":
        return session

    if "error" in session:
        return {
            "agent": "investigator",
            "session_id": session_id,
            "status": "session_not_found",
            "error": session["error"],
        }

    spoof_probability = float(spoof_probability)

    if spoof_probability >= 0.65:
        risk_level = "escalate_to_merchant"
    elif spoof_probability >= 0.35:
        risk_level = "soft_flag"
    else:
        risk_level = "clear"

    investigation = {
        "agent": "investigator",
        "session_id": session_id,
        "spoof_probability": spoof_probability,
        "risk_level": risk_level,
        "session_details": session,
    }

    # --------------------------------------------------------
    # Actual tool signature:
    #
    # submit_investigation_verdict(
    #     session_id,
    #     recommendation,
    #     explanation,
    #     confidence
    # )
    # --------------------------------------------------------

    verdict = _safe_call(
        submit_investigation_verdict,
        session_id=session_id,
        recommendation=risk_level,
        explanation=(
            "Gray-zone transaction reviewed by Investigator Agent."
        ),
        confidence=spoof_probability,
    )

    investigation["verdict_submission"] = verdict

    return investigation


# ============================================================
# 3. Evidence Compiler Agent
# ============================================================

def run_evidence_compiler(
    session_id: str,
) -> Dict[str, Any]:
    """
    Builds an evidence package for a suspicious transaction.
    """

    session = _safe_call(
        get_session_details,
        session_id,
    )

    if session.get("status") == "agent_error":
        return session

    if "error" in session:
        return {
            "agent": "evidence_compiler",
            "session_id": session_id,
            "status": "session_not_found",
            "error": session["error"],
        }

    evidence = {
        "agent": "evidence_compiler",
        "session_id": session_id,
        "session": session,
        "evidence": [],
    }

    fields = [
        "scan_time_delta",
        "ip_request_velocity",
        "user_agent_score",
        "bank_webhook_received",
        "status",
        "amount",
        "expected_amount",
        "settled_amount",
        "classifier_confidence",
        "device_fingerprint",
        "ip_address",
        "vpa",
        "flagged",
    ]

    for field in fields:
        if field in session:
            evidence["evidence"].append({
                "field": field,
                "value": session[field],
            })

    return evidence


# ============================================================
# 4. Ring Detector Agent
# ============================================================

def run_ring_detector(
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Looks for related suspicious sessions.

    This is advisory intelligence only.
    """

    # --------------------------------------------------------
    # Build a list of session IDs for clustering.
    #
    # The actual implementation expects:
    #
    # cluster_by_fingerprint_similarity(
    #     session_ids: list,
    #     min_cluster_size: int = 3
    # )
    # --------------------------------------------------------

    if session_id is not None:

        target_session = STORE["sessions"].get(session_id)

        if not target_session:
            return {
                "agent": "ring_detector",
                "session_id": session_id,
                "status": "session_not_found",
                "error": f"unknown session_id {session_id}",
            }

        target_fingerprint = target_session.get(
            "device_fingerprint"
        )

        related_session_ids = [
            sid
            for sid, session in STORE["sessions"].items()
            if session.get("device_fingerprint") == target_fingerprint
        ]

    else:
        related_session_ids = list(STORE["sessions"].keys())

    result = _safe_call(
        cluster_by_fingerprint_similarity,
        session_ids=related_session_ids,
        min_cluster_size=2,
    )

    return {
        "agent": "ring_detector",
        "session_id": session_id,
        "sessions_analyzed": len(related_session_ids),
        "result": result,
    }


# ============================================================
# 5. Merchant Copilot
# ============================================================

def run_merchant_copilot(
    merchant_id: str,
) -> Dict[str, Any]:
    """
    Generates merchant-facing defensive intelligence.

    Copilot does not automatically change payment state.
    """

    # get_flagged_sessions_batch() requires:
    #
    # start_time
    # end_time
    # min_confidence

    end_time = STORE.get(
        "_copilot_now",
        None,
    )

    if end_time is None:
        from datetime import datetime

        end_time = datetime.utcnow().isoformat()

    from datetime import datetime, timedelta

    try:
        end_dt = datetime.fromisoformat(
            end_time.replace("Z", "")
        )
    except Exception:
        end_dt = datetime.utcnow()

    start_dt = end_dt - timedelta(minutes=15)

    flagged = _safe_call(
        get_flagged_sessions_batch,
        start_time=start_dt.isoformat(),
        end_time=end_dt.isoformat(),
        min_confidence=0.5,
    )

    # Filter to requested merchant because the underlying
    # demo tool returns all flagged sessions.
    if (
        isinstance(flagged, dict)
        and flagged.get("status") != "agent_error"
    ):
        sessions = flagged.get("sessions", [])

        merchant_sessions = [
            s
            for s in sessions
            if s.get("merchant_id") == merchant_id
        ]

        flagged = {
            "count": len(merchant_sessions),
            "sessions": merchant_sessions,
        }

    return {
        "agent": "merchant_copilot",
        "merchant_id": merchant_id,
        "flagged_sessions": flagged,
        "recommendation": (
            "Review flagged sessions and investigate repeated "
            "device/IP patterns before taking operational action."
        ),
    }


# ============================================================
# 6. Red-Team Sandbox
# ============================================================

def run_red_team_sandbox(
    sandbox_merchant_id: str,
    strategy: str = "timing_jitter",
) -> Dict[str, Any]:
    """
    Runs defensive red-team analysis.

    HARD SAFETY FENCE:
    Only SANDBOX_* merchant IDs are allowed.
    """

    if not sandbox_merchant_id.startswith("SANDBOX_"):
        return {
            "status": "blocked",
            "agent": "red_team",
            "reason": (
                "Red-Team is sandbox-only. "
                "sandbox_merchant_id must start with SANDBOX_."
            ),
        }

    try:
        from agents.tool_implementations import (
            generate_spoof_attempt_pattern,
        )

        # Actual signature:
        #
        # generate_spoof_attempt_pattern(
        #     strategy,
        #     sandbox_merchant_id
        # )

        pattern = _safe_call(
            generate_spoof_attempt_pattern,
            strategy=strategy,
            sandbox_merchant_id=sandbox_merchant_id,
        )

        if (
            isinstance(pattern, dict)
            and pattern.get("status") == "agent_error"
        ):
            return {
                "agent": "red_team",
                "sandbox": True,
                "merchant_id": sandbox_merchant_id,
                "status": "analysis_error",
                "error": pattern.get("error"),
            }

        return {
            "agent": "red_team",
            "sandbox": True,
            "merchant_id": sandbox_merchant_id,
            "strategy": strategy,
            "result": pattern,
        }

    except ImportError:
        return {
            "agent": "red_team",
            "sandbox": True,
            "merchant_id": sandbox_merchant_id,
            "status": "not_available",
            "message": (
                "generate_spoof_attempt_pattern is not available "
                "in tool_implementations.py"
            ),
        }

    except Exception as exc:
        return {
            "agent": "red_team",
            "sandbox": True,
            "merchant_id": sandbox_merchant_id,
            "status": "analysis_error",
            "error": str(exc),
        }