# Fraud-Defense

This repository now includes a minimal reference implementation for QR spoof-payment defense with four core layers:

1. **Dynamic Web Router (No raw static UPI in QR):**
   - Creates signed, time-bound URLs (`https://kavach.pay/<session_id>?exp=...&sig=...`) instead of embedding `upi://pay?pa=...` in printed QR codes.
2. **Gateway Session Binding + Redirect:**
   - Validates session signature and expiry before generating the UPI deep-link.
   - Logs scan/session activity server-side.
3. **Fraud/Replay Risk Detector:**
   - Scores suspicious behavior using telemetry signals (scan latency, IP velocity, unfulfilled ratio).
4. **Settlement Signaling:**
   - Triggers merchant-side settlement only when bank webhook confirmation is received.

## Files

- `fraud_defense.py` — core implementation
- `test_fraud_defense.py` — focused behavioral tests

## Run tests

```bash
python -m unittest -v
```
