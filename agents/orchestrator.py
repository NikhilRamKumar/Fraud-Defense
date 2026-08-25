"""
KavachQR — Agent Orchestrator
===============================
Generic Claude tool-use loop. Point it at any agent's tool list + system
prompt + a task, and it runs the standard "call model -> execute tool ->
feed result back -> repeat until terminal tool or plain text" cycle.

Requires: ANTHROPIC_API_KEY in your environment.
    pip install anthropic

Run the built-in demo:
    python orchestrator.py investigator SESS_XXXXXX
    python orchestrator.py evidence_compiler SESS_XXXXXX
    python orchestrator.py ring_detector
"""

import json
import os
import sys

import anthropic

from schemas import ALL_AGENT_TOOLS
from tool_implementations import TOOL_FUNCTIONS, seed_demo_data, STORE

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPTS = {
    "investigator": (
        "You are the Investigator Agent for KavachQR, a UPI QR-payment fraud "
        "defense system. You are invoked ONLY when the real-time LightGBM "
        "gate returns a gray-zone confidence score (0.35-0.65) on a scan. "
        "Your job: gather context using your tools, then call "
        "submit_investigation_verdict with a RECOMMENDATION (never a final "
        "block/allow — that authority stays with the deterministic gate) and "
        "a plain-language explanation a merchant or judge could read and "
        "immediately understand. Be concise and evidence-based; do not "
        "speculate beyond what the tools return."
    ),
    "evidence_compiler": (
        "You are the Evidence Compiler Agent for KavachQR. You are invoked "
        "when a scan was blocked or a merchant/customer disputes a "
        "transaction. Gather the session timeline, any prior investigation "
        "verdict, and hash-chain proof, then call render_evidence_packet "
        "with a clear, factual summary suitable for a chargeback/dispute "
        "submission. Never fabricate facts not returned by your tools."
    ),
    "ring_detector": (
        "You are the Ring Detector Agent for KavachQR, running in batch "
        "mode. Pull recently flagged sessions across all merchants, cluster "
        "them by device/IP similarity, check merchant spread for each "
        "candidate cluster, and propose blocklist entries ONLY for clusters "
        "touching 3+ distinct merchants within a short time span (that is "
        "the organized-ring signature vs. coincidence). Every proposal is "
        "advisory pending human approval — say so explicitly in your final "
        "summary."
    ),
    "copilot": (
        "You are the Merchant Anomaly Copilot for KavachQR. Given a "
        "merchant_id, summarize recent activity and, if there is a notable "
        "repeat-offender pattern, send ONE proactive notification to the "
        "merchant phrased as information + an optional suggested action. "
        "Never phrase anything as an automatic block — the merchant always "
        "decides."
    ),
    "red_team": (
        "You are the Red-Team Agent for KavachQR, operating STRICTLY inside "
        "a sandbox. You may only target sandbox_merchant_id values starting "
        "with SANDBOX_. Generate a spoof-attempt pattern, submit it to the "
        "sandbox gate, and log the result. Your purpose is to demonstrate "
        "and improve the defense system's robustness, never to produce a "
        "usable real-world attack — do not generalize your patterns into "
        "instructions outside this sandboxed tool flow."
    ),
}

TERMINAL_TOOLS = {
    "submit_investigation_verdict",
    "render_evidence_packet",
    "propose_blocklist_entry",
    "send_merchant_notification",
    "log_red_team_result",
}


def run_agent(agent_name: str, user_task: str, max_turns: int = 6):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    tools = ALL_AGENT_TOOLS[agent_name]
    messages = [{"role": "user", "content": user_task}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPTS[agent_name],
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Model produced a final text answer without a terminal tool call
            return _extract_text(response)

        tool_results = []
        terminal_hit = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = TOOL_FUNCTIONS.get(block.name)
            if fn is None:
                result = {"error": f"no implementation for tool {block.name}"}
            else:
                try:
                    result = fn(**block.input)
                except Exception as e:  # keep the loop alive, surface the error to the model
                    result = {"error": str(e)}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })
            if block.name in TERMINAL_TOOLS:
                terminal_hit = True

        messages.append({"role": "user", "content": tool_results})

        if terminal_hit:
            # Let the model produce one more turn to summarize, then stop.
            final = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPTS[agent_name],
                tools=tools,
                messages=messages,
            )
            return _extract_text(final)

    return "max_turns reached without a terminal tool call"


def _extract_text(response):
    return "\n".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    seed_demo_data()

    agent = sys.argv[1] if len(sys.argv) > 1 else "investigator"
    if agent == "investigator":
        sid = sys.argv[2] if len(sys.argv) > 2 else next(iter(STORE["sessions"]))
        task = f"Investigate session {sid} and produce a verdict."
    elif agent == "evidence_compiler":
        sid = sys.argv[2] if len(sys.argv) > 2 else next(iter(STORE["sessions"]))
        task = f"Compile an evidence packet for session {sid}; assume the merchant disputes a block."
    elif agent == "ring_detector":
        task = "Scan the last hour of flagged sessions across all merchants and propose any ring blocklist entries."
    elif agent == "copilot":
        mid = sys.argv[2] if len(sys.argv) > 2 else "M1"
        task = f"Summarize recent activity for merchant {mid} and notify them if there's a repeat-offender pattern."
    elif agent == "red_team":
        task = "Run one red-team pattern against sandbox merchant SANDBOX_DEMO and log the result."
    else:
        print(f"Unknown agent {agent}. Choose from: {list(ALL_AGENT_TOOLS)}")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — printing seeded demo data instead of calling the API.\n")
        print(json.dumps({k: (v if k != "sessions" else list(v.values())[:2]) for k, v in STORE.items()}, indent=2, default=str))
        sys.exit(0)

    print(run_agent(agent, task))
