from datetime import datetime, timedelta, timezone
import unittest
from urllib.parse import urlparse

from fraud_defense import DynamicQRRouter, FraudDetector


class DynamicQRRouterTests(unittest.TestCase):
    def test_session_url_has_no_direct_upi_parameters(self) -> None:
        router = DynamicQRRouter(secret_key="secret")
        url = router.create_session("store-1", 2599)
        self.assertTrue(url.startswith("https://kavach.pay/"))
        self.assertNotIn("upi://pay", url)
        self.assertNotIn("pa=", url)

    def test_gateway_redirect_only_after_signature_validation(self) -> None:
        router = DynamicQRRouter(secret_key="secret")
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        url = router.create_session("store-1", 2599, now=created_at)

        upi_link = router.resolve_gateway_redirect(
            url,
            source_ip="10.0.0.1",
            scan_latency_ms=340,
            now=created_at + timedelta(seconds=10),
        )
        self.assertTrue(upi_link.startswith("upi://pay?"))

        tampered = url.replace("sig=", "sig=bad")
        with self.assertRaises(ValueError):
            router.resolve_gateway_redirect(
                tampered,
                source_ip="10.0.0.1",
                scan_latency_ms=30,
                now=created_at + timedelta(seconds=10),
            )

    def test_settlement_signal_only_after_webhook_confirmed_payment(self) -> None:
        router = DynamicQRRouter(secret_key="secret")
        url = router.create_session("store-1", 1000)
        session_id = urlparse(url).path.strip("/")

        pending = router.webhook_settlement(session_id, settled=False)
        self.assertFalse(pending.triggered)

        settled = router.webhook_settlement(session_id, settled=True)
        self.assertTrue(settled.triggered)
        self.assertEqual(settled.reason, "bank_confirmed")


class FraudDetectorTests(unittest.TestCase):
    def test_detector_flags_replay_like_pattern(self) -> None:
        detector = FraudDetector()
        self.assertTrue(
            detector.is_suspicious(
                scan_latency_ms=20,
                ip_velocity_per_minute=30,
                unfulfilled_ratio=0.9,
            )
        )

    def test_detector_allows_legit_pattern(self) -> None:
        detector = FraudDetector()
        self.assertFalse(
            detector.is_suspicious(
                scan_latency_ms=500,
                ip_velocity_per_minute=1,
                unfulfilled_ratio=0.05,
            )
        )


if __name__ == "__main__":
    unittest.main()
