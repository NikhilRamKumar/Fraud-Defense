"""
KavachQR — Agent Tool Schemas
==============================
Anthropic `tools` param definitions for every agent in the system.

Design principle: each agent gets the SMALLEST tool set that lets it do its
job — this keeps latency down, keeps the agent auditable ("it could only
have looked at X, Y, Z"), and keeps every agent strictly read-only /
defense-only (no tool here can send money, block a real user permanently,
or take an irreversible action without a human-visible log entry).

Import the relevant list into your agent's system prompt / API call:
    from agents.schemas import INVESTIGATOR_TOOLS, EVIDENCE_COMPILER_TOOLS, ...
"""

# ---------------------------------------------------------------------------
# 1. INVESTIGATOR AGENT
# Triggered only on gray-zone LightGBM confidence scores (e.g. 0.35-0.65).
# Job: gather context, produce a human-readable explanation + a recommendation
# (never a final block/allow decision — that stays with the deterministic gate).
# ---------------------------------------------------------------------------
INVESTIGATOR_TOOLS = [
    {
        "name": "get_session_details",
        "description": (
            "Fetch full metadata for a single QR scan session: timestamps, "
            "scan_time_delta, ip_request_velocity, user_agent_score, "
            "bank_webhook_received, classifier confidence, and raw request headers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "e.g. SESS_A1B2C3"}
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "get_device_history",
        "description": (
            "Look up prior scan/session history for a device fingerprint or "
            "IP address across this merchant only (not cross-merchant — that "
            "is the Ring Detector's job). Returns counts of past genuine vs "
            "blocked attempts and recency."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "device_fingerprint": {"type": "string"},
                "ip_address": {"type": "string"},
                "lookback_hours": {"type": "integer", "default": 72},
            },
            "required": [],
        },
    },
    {
        "name": "get_merchant_baseline",
        "description": (
            "Return this merchant's normal transaction pattern: typical "
            "ticket size range, typical scans/hour, typical genuine:blocked "
            "ratio. Used to judge whether current session is anomalous "
            "relative to THIS merchant, not a global average."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"merchant_id": {"type": "string"}},
            "required": ["merchant_id"],
        },
    },
    {
        "name": "validate_upi_handle_format",
        "description": (
            "Static, offline check that a UPI handle / VPA conforms to "
            "NPCI-style syntax (handle@psp) and that the PSP suffix is a "
            "known bank/PSP code. Does NOT call any external bank API — "
            "pure format/allowlist validation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"vpa": {"type": "string"}},
            "required": ["vpa"],
        },
    },
    {
        "name": "submit_investigation_verdict",
        "description": (
            "Terminal tool for this agent. Records the agent's explanation "
            "and a RECOMMENDATION ONLY (escalate / soft-flag / clear). This "
            "never directly blocks or allows a transaction — it writes to "
            "an audit log that a human or the deterministic gate can act on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "recommendation": {
                    "type": "string",
                    "enum": ["escalate_to_merchant", "soft_flag", "clear"],
                },
                "explanation": {
                    "type": "string",
                    "description": "Plain-language reasoning, shown to merchant/judges.",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["session_id", "recommendation", "explanation", "confidence"],
        },
    },
]

# ---------------------------------------------------------------------------
# 2. EVIDENCE COMPILER AGENT
# Triggered on: a scan blocked by the gate, OR a merchant/customer dispute.
# Job: assemble a structured, tamper-evident evidence packet. Never edits
# underlying logs — read-only over the session/audit store, write-only to
# a NEW evidence_packets table.
# ---------------------------------------------------------------------------
EVIDENCE_COMPILER_TOOLS = [
    {
        "name": "get_session_timeline",
        "description": (
            "Return the ordered event timeline for a session: QR generated, "
            "scanned, ML decision, webhook received/absent, merchant status "
            "update — each with timestamp and source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "get_investigation_record",
        "description": (
            "Fetch the Investigator Agent's prior verdict for this session, "
            "if one exists (explanation, recommendation, confidence)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "get_hash_chain_proof",
        "description": (
            "Return the hash-chain segment (this session's log hash + "
            "previous hash) proving the session record has not been altered "
            "after the fact. Used as tamper-evidence in disputes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "render_evidence_packet",
        "description": (
            "Terminal tool. Compiles gathered facts into a structured "
            "dispute-response document (JSON + human-readable summary) "
            "formatted for a chargeback/dispute submission. Does not send "
            "it anywhere — returns it for merchant review/export."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "dispute_reason": {
                    "type": "string",
                    "description": "e.g. 'customer claims payment failed but was blocked as spoof'",
                },
                "summary": {"type": "string"},
                "timeline": {"type": "array", "items": {"type": "object"}},
                "verdict_confidence": {"type": "number"},
            },
            "required": ["session_id", "summary", "timeline"],
        },
    },
]

# ---------------------------------------------------------------------------
# 3. RING DETECTOR AGENT (batch / offline)
# Job: cluster flagged spoof attempts ACROSS merchants to surface organized
# fraud rings. Read-only over aggregated, anonymized flagged-session data.
# Output feeds a shared blocklist — never auto-bans, only proposes.
# ---------------------------------------------------------------------------
RING_DETECTOR_TOOLS = [
    {
        "name": "get_flagged_sessions_batch",
        "description": (
            "Return all sessions flagged as spoof/blocked across all "
            "merchants within a time window, with device fingerprint, IP "
            "block (/24), and timing metadata. PII (VPA, phone) excluded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {"type": "string", "description": "ISO 8601"},
                "end_time": {"type": "string", "description": "ISO 8601"},
                "min_confidence": {"type": "number", "default": 0.5},
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "cluster_by_fingerprint_similarity",
        "description": (
            "Run clustering (device fingerprint, IP /24 block, and inter-"
            "attempt timing) over a batch of flagged sessions and return "
            "candidate clusters with a cohesion score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_ids": {"type": "array", "items": {"type": "string"}},
                "min_cluster_size": {"type": "integer", "default": 3},
            },
            "required": ["session_ids"],
        },
    },
    {
        "name": "get_cluster_merchant_spread",
        "description": (
            "For a candidate cluster, return how many distinct merchants it "
            "touched and over what time span — used to distinguish an "
            "organized ring (many merchants, short span) from coincidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"cluster_id": {"type": "string"}},
            "required": ["cluster_id"],
        },
    },
    {
        "name": "propose_blocklist_entry",
        "description": (
            "Terminal tool. Proposes (does NOT enact) a shared-blocklist "
            "entry for a device fingerprint / IP block, with supporting "
            "evidence. Requires human or downstream-policy approval before "
            "it affects the live Verifier Agent gate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_id": {"type": "string"},
                "fingerprint_or_ip": {"type": "string"},
                "evidence_summary": {"type": "string"},
                "merchant_count": {"type": "integer"},
                "confidence": {"type": "number"},
            },
            "required": ["cluster_id", "fingerprint_or_ip", "evidence_summary", "confidence"],
        },
    },
]

# ---------------------------------------------------------------------------
# 4. MERCHANT ANOMALY COPILOT (conversational, real-time-adjacent)
# Job: proactively surface patterns to the merchant in plain language and
# answer merchant questions about their own transaction stream.
# ---------------------------------------------------------------------------
COPILOT_TOOLS = [
    {
        "name": "get_recent_activity_summary",
        "description": (
            "Summarized counts for this merchant over a recent window: "
            "genuine settlements, blocked attempts, gray-zone escalations, "
            "grouped by device/IP where repeated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "merchant_id": {"type": "string"},
                "window_minutes": {"type": "integer", "default": 15},
            },
            "required": ["merchant_id"],
        },
    },
    {
        "name": "get_investigation_record",  # shared with Evidence Compiler
        "description": "Fetch Investigator Agent verdict for a session, if any.",
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "send_merchant_notification",
        "description": (
            "Terminal tool. Sends a plain-language proactive alert to the "
            "merchant app UI (push/toast), e.g. '3 blocked attempts, same "
            "device, last 10 min — want to flag this customer?'. Never "
            "auto-blocks; always phrased as information + optional action "
            "for the merchant to take."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "merchant_id": {"type": "string"},
                "message": {"type": "string"},
                "suggested_action": {
                    "type": "string",
                    "enum": ["none", "flag_customer", "review_evidence", "contact_support"],
                },
            },
            "required": ["merchant_id", "message", "suggested_action"],
        },
    },
]

# ---------------------------------------------------------------------------
# 5. RED-TEAM AGENT (adversarial self-test — sandboxed, demo-only)
# Job: generate NEW spoof-attempt patterns against a SANDBOX instance of the
# gateway only. Never targets production sessions or real bank webhooks.
# This is the one agent that could look "offense-capable" so it is fenced
# hard: every tool is scoped to a sandbox flag and nothing here can reach
# real merchant/customer data.
# ---------------------------------------------------------------------------
RED_TEAM_TOOLS = [
    {
        "name": "generate_spoof_attempt_pattern",
        "description": (
            "SANDBOX ONLY. Generate a synthetic scan-request pattern "
            "(timing, header, velocity variations) to test the Verifier "
            "Agent's robustness. Operates purely on synthetic session IDs "
            "prefixed SANDBOX_ — cannot reference real session IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["timing_jitter", "header_spoof", "velocity_burst", "replay_stale_token"],
                },
                "sandbox_merchant_id": {
                    "type": "string",
                    "description": "Must start with SANDBOX_",
                },
            },
            "required": ["strategy", "sandbox_merchant_id"],
        },
    },
    {
        "name": "submit_to_sandbox_gate",
        "description": (
            "Sends a generated pattern to the SANDBOX instance of the "
            "Verifier Agent's LightGBM gate and returns its decision + "
            "confidence + latency. Sandbox only — no production route "
            "exists for this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"pattern_id": {"type": "string"}},
            "required": ["pattern_id"],
        },
    },
    {
        "name": "log_red_team_result",
        "description": (
            "Terminal tool. Records whether the sandbox gate caught the "
            "generated pattern, for the live demo dashboard and for "
            "(optional, human-approved) threshold-tuning suggestions. "
            "Never modifies the production model directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern_id": {"type": "string"},
                "strategy": {"type": "string"},
                "caught": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": ["pattern_id", "strategy", "caught"],
        },
    },
]

# Convenience: everything in one place for an orchestrator that routes by agent name
ALL_AGENT_TOOLS = {
    "investigator": INVESTIGATOR_TOOLS,
    "evidence_compiler": EVIDENCE_COMPILER_TOOLS,
    "ring_detector": RING_DETECTOR_TOOLS,
    "copilot": COPILOT_TOOLS,
    "red_team": RED_TEAM_TOOLS,
}
