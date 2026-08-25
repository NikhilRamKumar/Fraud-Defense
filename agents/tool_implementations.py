"""
KavachQR — Tool Implementations
=================================
Real Python functions behind every tool name declared in schemas.py.

For the hackathon build these read/write an in-memory store (`STORE`) that
you seed from your existing FastAPI session dict / qr_sessions.csv. Swap
`STORE` for your real DB/session table later — the function signatures
(tool_name -> dict in, dict out) don't need to change.

Usage from your agent loop:
    from agents.tool_implementations import TOOL_FUNCTIONS
    result = TOOL_FUNCTIONS[tool_name](**tool_input)
"""

import hashlib
import random
import statistics
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Mock in-memory store — replace with real session/audit DB in production.
# ---------------------------------------------------------------------------
STORE = {
    "sessions": {},          # session_id -> dict
    "audit_log": [],         # append-only list of events, hash-chained
    "investigations": {},    # session_id -> verdict dict
    "evidence_packets": {},  # session_id -> packet dict
    "notifications": [],     # merchant notification log
    "blocklist_proposals": [],
    "red_team_log": [],
}


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


def _hash_record(record: dict, prev_hash: str) -> str:
    payload = f"{prev_hash}|{sorted(record.items())}".encode()
    return hashlib.sha256(payload).hexdigest()


def seed_demo_data(n_sessions: int = 40, n_merchants: int = 5):
    """Populate STORE with plausible synthetic sessions + hash-chained audit
    log, so every tool below returns something real during a demo."""
    prev_hash = "GENESIS"
    for i in range(n_sessions):
        sid = f"SESS_{uuid.uuid4().hex[:8].upper()}"
        merchant_id = f"M{random.randint(1, n_merchants)}"
        is_spoof = random.random() < 0.15
        record = {
            "session_id": sid,
            "merchant_id": merchant_id,
            "created_at": (datetime.utcnow() - timedelta(minutes=random.randint(0, 600))).isoformat(),
            "scan_time_delta": round(random.uniform(0.5, 45.0), 2),
            "ip_request_velocity": random.randint(1, 12) if not is_spoof else random.randint(8, 40),
            "user_agent_score": round(random.uniform(0.7, 1.0), 2) if not is_spoof else round(random.uniform(0.0, 0.4), 2),
            "bank_webhook_received": (not is_spoof) and random.random() < 0.95,
            "device_fingerprint": f"FP{random.randint(1, 12)}",
            "ip_address": f"103.21.{random.randint(0,255)}.{random.randint(0,255)}",
            "vpa": f"vendor{merchant_id}@okhdfcbank" if not is_spoof else "vendorX@fakepsp",
            "amount": random.choice([10, 20, 50, 100]),
            "classifier_confidence": round(random.uniform(0.85, 0.99), 3) if not is_spoof else round(random.uniform(0.4, 0.95), 3),
            "flagged": is_spoof,
        }
        STORE["sessions"][sid] = record

        for event_type in ["qr_generated", "scanned", "ml_decision", "webhook_received", "merchant_status_update"]:
            event = {
                "session_id": sid,
                "event": event_type,
                "timestamp": _now_iso(),
                "source": "gateway" if event_type != "webhook_received" else "bank",
            }
            h = _hash_record(event, prev_hash)
            event["hash"] = h
            event["prev_hash"] = prev_hash
            prev_hash = h
            STORE["audit_log"].append(event)

# ---------------------------------------------------------------------------
# FASTAPI -> AGENT STORE SYNC
# ---------------------------------------------------------------------------
def sync_fastapi_session(
    session_id: str,
    session_data: dict,
):
    """
    Synchronize a FastAPI session into the agent-side STORE.

    The agent layer only receives a copy of the session.
    It does not make the final payment decision.
    """

    if not session_id:
        raise ValueError("session_id is required")

    if not isinstance(session_data, dict):
        raise TypeError("session_data must be a dictionary")

    session = dict(session_data)

    # Ensure the session ID exists inside the stored record.
    session["session_id"] = session_id

    # Store the session for agent analysis.
    STORE["sessions"][session_id] = session

    return {
        "session_id": session_id,
        "stored": True,
        "merchant_id": session.get("merchant_id"),
        "flagged": session.get("flagged", False),
    }
# ---------------------------------------------------------------------------
# INVESTIGATOR AGENT tools
# ---------------------------------------------------------------------------
def get_session_details(session_id: str):
    s = STORE["sessions"].get(session_id)
    if not s:
        return {"error": f"unknown session_id {session_id}"}
    return s


def get_device_history(device_fingerprint: str = None, ip_address: str = None, lookback_hours: int = 72):
    matches = [
        s for s in STORE["sessions"].values()
        if (device_fingerprint and s["device_fingerprint"] == device_fingerprint)
        or (ip_address and s["ip_address"] == ip_address)
    ]
    genuine = sum(1 for m in matches if not m["flagged"])
    blocked = sum(1 for m in matches if m["flagged"])
    return {
        "total_sessions": len(matches),
        "genuine_count": genuine,
        "blocked_count": blocked,
        "most_recent": max((m["created_at"] for m in matches), default=None),
    }


def get_merchant_baseline(merchant_id: str):
    matches = [s for s in STORE["sessions"].values() if s["merchant_id"] == merchant_id]
    if not matches:
        return {"error": f"no data for merchant {merchant_id}"}
    amounts = [m["amount"] for m in matches]
    flagged_ratio = sum(1 for m in matches if m["flagged"]) / len(matches)
    return {
        "merchant_id": merchant_id,
        "avg_ticket_size": round(statistics.mean(amounts), 2),
        "ticket_size_range": [min(amounts), max(amounts)],
        "typical_flagged_ratio": round(flagged_ratio, 3),
        "sample_size": len(matches),
    }


KNOWN_PSP_SUFFIXES = {"okhdfcbank", "oksbi", "okaxis", "okicici", "ybl", "paytm", "apl"}


def validate_upi_handle_format(vpa: str):
    if "@" not in vpa:
        return {"valid": False, "reason": "missing @ separator"}
    handle, _, suffix = vpa.partition("@")
    valid_suffix = suffix.lower() in KNOWN_PSP_SUFFIXES
    return {
        "valid": bool(handle) and valid_suffix,
        "handle": handle,
        "suffix": suffix,
        "known_psp_suffix": valid_suffix,
    }


def submit_investigation_verdict(session_id: str, recommendation: str, explanation: str, confidence: float):
    verdict = {
        "session_id": session_id,
        "recommendation": recommendation,
        "explanation": explanation,
        "confidence": confidence,
        "recorded_at": _now_iso(),
    }
    STORE["investigations"][session_id] = verdict
    return {"status": "recorded", "verdict": verdict}


# ---------------------------------------------------------------------------
# EVIDENCE COMPILER AGENT tools
# ---------------------------------------------------------------------------
def get_session_timeline(session_id: str):
    events = [e for e in STORE["audit_log"] if e["session_id"] == session_id]
    return {"session_id": session_id, "timeline": events}


def get_investigation_record(session_id: str):
    return STORE["investigations"].get(session_id, {"status": "no_investigation_on_file"})


def get_hash_chain_proof(session_id: str):
    events = [e for e in STORE["audit_log"] if e["session_id"] == session_id]
    if not events:
        return {"error": "no audit trail for session"}
    return {
        "session_id": session_id,
        "first_hash": events[0]["hash"],
        "last_hash": events[-1]["hash"],
        "chain_length": len(events),
        "tamper_evident": True,
    }


def render_evidence_packet(session_id: str, summary: str, timeline: list, dispute_reason: str = "", verdict_confidence: float = None):
    packet = {
        "session_id": session_id,
        "dispute_reason": dispute_reason,
        "summary": summary,
        "timeline": timeline,
        "verdict_confidence": verdict_confidence,
        "hash_proof": get_hash_chain_proof(session_id),
        "generated_at": _now_iso(),
        "packet_id": f"EVID_{uuid.uuid4().hex[:10].upper()}",
    }
    STORE["evidence_packets"][session_id] = packet
    return packet


# ---------------------------------------------------------------------------
# RING DETECTOR AGENT tools
# ---------------------------------------------------------------------------
def get_flagged_sessions_batch(start_time: str, end_time: str, min_confidence: float = 0.5):
    # For the demo, ignore exact time parsing and just return flagged sessions.
    flagged = [
        {
            "session_id": s["session_id"],
            "merchant_id": s["merchant_id"],
            "device_fingerprint": s["device_fingerprint"],
            "ip_block": ".".join(s["ip_address"].split(".")[:3]) + ".0/24",
            "created_at": s["created_at"],
            "classifier_confidence": s["classifier_confidence"],
        }
        for s in STORE["sessions"].values()
        if s["flagged"] and s["classifier_confidence"] >= min_confidence
    ]
    return {"count": len(flagged), "sessions": flagged}


def cluster_by_fingerprint_similarity(session_ids: list, min_cluster_size: int = 3):
    groups = defaultdict(list)
    for sid in session_ids:
        s = STORE["sessions"].get(sid)
        if not s:
            continue
        groups[s["device_fingerprint"]].append(sid)

    clusters = []
    for fp, sids in groups.items():
        if len(sids) >= min_cluster_size:
            clusters.append({
                "cluster_id": f"CLUS_{fp}",
                "fingerprint": fp,
                "session_ids": sids,
                "cohesion_score": round(min(1.0, len(sids) / 10), 2),
            })
    return {"clusters": clusters}


def get_cluster_merchant_spread(cluster_id: str):
    fp = cluster_id.replace("CLUS_", "")
    matches = [s for s in STORE["sessions"].values() if s["device_fingerprint"] == fp]
    merchants = {m["merchant_id"] for m in matches}
    times = sorted(m["created_at"] for m in matches)
    return {
        "cluster_id": cluster_id,
        "distinct_merchants": len(merchants),
        "merchant_ids": list(merchants),
        "time_span_first": times[0] if times else None,
        "time_span_last": times[-1] if times else None,
    }


def propose_blocklist_entry(cluster_id: str, fingerprint_or_ip: str, evidence_summary: str, confidence: float, merchant_count: int = None):
    proposal = {
        "cluster_id": cluster_id,
        "fingerprint_or_ip": fingerprint_or_ip,
        "evidence_summary": evidence_summary,
        "merchant_count": merchant_count,
        "confidence": confidence,
        "status": "PENDING_HUMAN_APPROVAL",
        "proposed_at": _now_iso(),
    }
    STORE["blocklist_proposals"].append(proposal)
    return proposal


# ---------------------------------------------------------------------------
# COPILOT tools
# ---------------------------------------------------------------------------
def get_recent_activity_summary(merchant_id: str, window_minutes: int = 15):
    matches = [s for s in STORE["sessions"].values() if s["merchant_id"] == merchant_id]
    genuine = sum(1 for m in matches if not m["flagged"])
    blocked = sum(1 for m in matches if m["flagged"])
    by_device = defaultdict(int)
    for m in matches:
        if m["flagged"]:
            by_device[m["device_fingerprint"]] += 1
    repeated = {k: v for k, v in by_device.items() if v >= 2}
    return {
        "merchant_id": merchant_id,
        "window_minutes": window_minutes,
        "genuine_count": genuine,
        "blocked_count": blocked,
        "repeat_offenders": repeated,
    }


def send_merchant_notification(merchant_id: str, message: str, suggested_action: str = "none"):
    note = {
        "merchant_id": merchant_id,
        "message": message,
        "suggested_action": suggested_action,
        "sent_at": _now_iso(),
    }
    STORE["notifications"].append(note)
    return {"status": "sent", "notification": note}


# ---------------------------------------------------------------------------
# RED-TEAM AGENT tools (sandbox-fenced)
# ---------------------------------------------------------------------------
def generate_spoof_attempt_pattern(strategy: str, sandbox_merchant_id: str):
    if not sandbox_merchant_id.startswith("SANDBOX_"):
        return {"error": "sandbox_merchant_id must start with SANDBOX_ — refusing non-sandbox target"}
    pattern_id = f"PAT_{uuid.uuid4().hex[:8]}"
    pattern = {
        "pattern_id": pattern_id,
        "strategy": strategy,
        "sandbox_merchant_id": sandbox_merchant_id,
        "generated_at": _now_iso(),
    }
    STORE.setdefault("red_team_patterns", {})[pattern_id] = pattern
    return pattern


def submit_to_sandbox_gate(pattern_id: str):
    pattern = STORE.get("red_team_patterns", {}).get(pattern_id)
    if not pattern:
        return {"error": "unknown pattern_id"}
    # Mock decision: simple synthetic gate reacting to strategy type
    caught_prob = {
        "timing_jitter": 0.6,
        "header_spoof": 0.85,
        "velocity_burst": 0.9,
        "replay_stale_token": 0.97,
    }.get(pattern["strategy"], 0.7)
    caught = random.random() < caught_prob
    return {
        "pattern_id": pattern_id,
        "caught": caught,
        "confidence": round(random.uniform(0.6, 0.99), 3),
        "latency_ms": round(random.uniform(8, 45), 1),
    }


def log_red_team_result(pattern_id: str, strategy: str, caught: bool, notes: str = ""):
    entry = {
        "pattern_id": pattern_id,
        "strategy": strategy,
        "caught": caught,
        "notes": notes,
        "logged_at": _now_iso(),
    }
    STORE["red_team_log"].append(entry)
    return entry


# ---------------------------------------------------------------------------
# Registry: tool_name -> callable (used by the orchestrator's dispatch loop)
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    # investigator
    "get_session_details": get_session_details,
    "get_device_history": get_device_history,
    "get_merchant_baseline": get_merchant_baseline,
    "validate_upi_handle_format": validate_upi_handle_format,
    "submit_investigation_verdict": submit_investigation_verdict,
    # evidence compiler
    "get_session_timeline": get_session_timeline,
    "get_investigation_record": get_investigation_record,
    "get_hash_chain_proof": get_hash_chain_proof,
    "render_evidence_packet": render_evidence_packet,
    # ring detector
    "get_flagged_sessions_batch": get_flagged_sessions_batch,
    "cluster_by_fingerprint_similarity": cluster_by_fingerprint_similarity,
    "get_cluster_merchant_spread": get_cluster_merchant_spread,
    "propose_blocklist_entry": propose_blocklist_entry,
    # copilot
    "get_recent_activity_summary": get_recent_activity_summary,
    "send_merchant_notification": send_merchant_notification,
    # red team
    "generate_spoof_attempt_pattern": generate_spoof_attempt_pattern,
    "submit_to_sandbox_gate": submit_to_sandbox_gate,
    "log_red_team_result": log_red_team_result,
}
