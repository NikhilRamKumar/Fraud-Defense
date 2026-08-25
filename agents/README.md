# KavachQR — Agent Tool Scaffold

Drop-in scaffold for the five agents discussed: Investigator, Evidence
Compiler, Ring Detector, Merchant Copilot, Red-Team (sandboxed).

## Files

- `agents/schemas.py` — Anthropic `tools` JSON schemas, one list per agent
  (`INVESTIGATOR_TOOLS`, `EVIDENCE_COMPILER_TOOLS`, etc.), plus
  `ALL_AGENT_TOOLS` keyed by agent name.
- `agents/tool_implementations.py` — the real Python functions behind every
  tool name, backed by an in-memory `STORE` (sessions, hash-chained audit
  log, investigations, evidence packets, blocklist proposals). `seed_demo_data()`
  fills it with plausible synthetic sessions so every tool has something to
  return immediately.
- `agents/orchestrator.py` — a generic Claude tool-use loop: call model →
  execute whichever tool it picks → feed the result back → repeat until a
  **terminal tool** (the one that actually records a verdict/packet/proposal)
  fires, then let the model summarize once more and stop.

## Try it now

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
cd agents
python orchestrator.py investigator SESS_XXXXXX
python orchestrator.py evidence_compiler SESS_XXXXXX
python orchestrator.py ring_detector
python orchestrator.py copilot M1
python orchestrator.py red_team
```

Without an API key it still seeds and prints the mock store, so you can see
the data shape before wiring in the model.

## Wiring into your existing FastAPI backend (`backend/main.py`)

1. Replace `STORE` in `tool_implementations.py` with reads/writes against
   your real session dict / DB — the function signatures don't change.
2. Call `run_agent("investigator", task)` from inside `/scan/{session_id}`
   **only** when the LightGBM confidence lands in the gray zone
   (e.g. 0.35–0.65) — keep the fast path fast.
3. Call `run_agent("evidence_compiler", task)` from a new
   `POST /dispute/{session_id}` endpoint.
4. Run `run_agent("ring_detector", task)` on a schedule (cron / background
   task), not per-request.
5. `red_team` only ever targets `sandbox_merchant_id` values prefixed
   `SANDBOX_` — `generate_spoof_attempt_pattern` hard-refuses anything else.
   Keep it pointed at a sandbox deployment, never production, for the demo.

## Design notes for judges

- Every agent's **terminal tool is advisory, not enactive**: verdicts,
  packets, and blocklist proposals all require the deterministic gate or a
  human to act on them. Nothing here can silently block a real payment or
  ban a real user on its own.
- The Red-Team agent is fenced at the tool-schema and implementation level
  (`SANDBOX_` prefix check) so it can't be pointed at production — this is
  what keeps the whole system defense-only per the track's disqualification
  rule.
- The hash-chain (`get_hash_chain_proof`) gives the Evidence Compiler a
  tamper-evidence story, not just "trust our logs."
