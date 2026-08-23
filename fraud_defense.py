from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs


@dataclass
class Session:
    session_id: str
    merchant_id: str
    amount_paise: int
    expires_at: datetime
    status: str = "created"
    scans: List["ScanTelemetry"] = field(default_factory=list)


@dataclass
class ScanTelemetry:
    scanned_at: datetime
    source_ip: str
    scan_latency_ms: int


@dataclass
class SettlementSignal:
    session_id: str
    triggered: bool
    reason: str


class DynamicQRRouter:
    """Creates and verifies time-bound, signed session URLs."""

    def __init__(self, secret_key: str, base_url: str = "https://kavach.pay") -> None:
        self._secret_key = secret_key.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self._sessions: Dict[str, Session] = {}
        self._scan_log: List[str] = []

    def create_session(
        self,
        merchant_id: str,
        amount_paise: int,
        ttl_seconds: int = 90,
        now: Optional[datetime] = None,
    ) -> str:
        now = now or datetime.now(timezone.utc)
        session_id = secrets.token_urlsafe(12)
        expires_at = now + timedelta(seconds=ttl_seconds)
        session = Session(
            session_id=session_id,
            merchant_id=merchant_id,
            amount_paise=amount_paise,
            expires_at=expires_at,
        )
        self._sessions[session_id] = session

        signature = self._sign(
            session_id=session_id,
            merchant_id=merchant_id,
            amount_paise=amount_paise,
            expires_at=expires_at,
        )
        query = urlencode({"exp": int(expires_at.timestamp()), "sig": signature})
        return f"{self.base_url}/{session_id}?{query}"

    def resolve_gateway_redirect(
        self,
        session_url: str,
        source_ip: str,
        scan_latency_ms: int,
        now: Optional[datetime] = None,
    ) -> str:
        now = now or datetime.now(timezone.utc)
        session = self._verify_url(session_url, now)
        session.scans.append(
            ScanTelemetry(scanned_at=now, source_ip=source_ip, scan_latency_ms=scan_latency_ms)
        )
        session.status = "scanned"
        self._scan_log.append(session.session_id)
        params = urlencode(
            {
                "pa": f"merchant-{session.merchant_id}@upi",
                "am": f"{session.amount_paise / 100:.2f}",
                "tn": session.session_id,
            }
        )
        return f"upi://pay?{params}"

    def webhook_settlement(self, session_id: str, settled: bool) -> SettlementSignal:
        session = self._sessions.get(session_id)
        if not session:
            return SettlementSignal(session_id=session_id, triggered=False, reason="unknown_session")
        if settled:
            session.status = "settled"
            return SettlementSignal(session_id=session_id, triggered=True, reason="bank_confirmed")
        session.status = "failed"
        return SettlementSignal(session_id=session_id, triggered=False, reason="not_settled")

    def session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    @property
    def scan_log(self) -> List[str]:
        return list(self._scan_log)

    def _verify_url(self, session_url: str, now: datetime) -> Session:
        parsed = urlparse(session_url)
        session_id = parsed.path.strip("/")
        query = parse_qs(parsed.query)
        exp = int(query["exp"][0])
        sig = query["sig"][0]

        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("invalid_session")
        if now.timestamp() > exp or now > session.expires_at:
            raise ValueError("expired_session")

        expected = self._sign(
            session_id=session.session_id,
            merchant_id=session.merchant_id,
            amount_paise=session.amount_paise,
            expires_at=session.expires_at,
        )
        if not hmac.compare_digest(sig, expected):
            raise ValueError("invalid_signature")
        return session

    def _sign(
        self,
        *,
        session_id: str,
        merchant_id: str,
        amount_paise: int,
        expires_at: datetime,
    ) -> str:
        payload = f"{session_id}|{merchant_id}|{amount_paise}|{int(expires_at.timestamp())}".encode(
            "utf-8"
        )
        return hmac.new(self._secret_key, payload, hashlib.sha256).hexdigest()


class FraudDetector:
    """Lightweight replay/fraud detector over scan telemetry."""

    def score(
        self,
        *,
        scan_latency_ms: int,
        ip_velocity_per_minute: int,
        unfulfilled_ratio: float,
    ) -> float:
        latency_component = max(0.0, min(1.0, (150 - scan_latency_ms) / 150))
        velocity_component = max(0.0, min(1.0, ip_velocity_per_minute / 20))
        unfulfilled_component = max(0.0, min(1.0, unfulfilled_ratio))
        return round((0.35 * latency_component) + (0.35 * velocity_component) + (0.3 * unfulfilled_component), 4)

    def is_suspicious(
        self,
        *,
        scan_latency_ms: int,
        ip_velocity_per_minute: int,
        unfulfilled_ratio: float,
        threshold: float = 0.6,
    ) -> bool:
        return self.score(
            scan_latency_ms=scan_latency_ms,
            ip_velocity_per_minute=ip_velocity_per_minute,
            unfulfilled_ratio=unfulfilled_ratio,
        ) >= threshold
