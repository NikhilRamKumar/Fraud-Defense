# Fraud-Defense
Small-to-medium retail merchants (Kirana stores, food stalls, transit hubs) frequently fall victim to fake payment confirmation scams. During high-traffic rush hours, merchants cannot verify incoming ledger notifications on their personal phones or listen for audio cues. Fraudsters exploit this by scanning static counter QRs .

## Run the prototype

```powershell
python -m pip install -r requirements.txt
$env:KAVACH_BASE_URL="https://your-ngrok-domain.ngrok-free.app"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `frontend/index.html` as the merchant console. Generate a session, then open its Customer Display link. The QR contains only `/scan/{session_id}`. For LAN testing, set `KAVACH_BASE_URL` to `http://192.168.x.x:8000`; for a physical phone, use the laptop's reachable LAN address or an HTTPS ngrok URL, never a loopback address.

Merchant setup uses `POST /merchant/register` followed by `POST /merchant/{merchant_id}/verify`. Verification is explicitly a demo-only bank simulation and stores only masked/display-safe account data. Sessions are server-created, single-use, and expire after 30 seconds. Dashboard events are delivered through `/ws/merchant/{merchant_id}`.
