"""Session-level audit logging for FaceTrust authentication events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SessionEvent:
    """Structured authentication event written to the session audit log."""

    timestamp: str
    person_name: str
    result: str
    confidence: float
    blink_detected: bool
    motion_detected: bool


class SessionLogger:
    """Append authentication events to a per-run CSV file with debounce."""

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.logs_dir / f"session_{run_stamp}.csv"
        self._last_event_key: Optional[tuple[str, str, float, bool, bool]] = None
        self._last_event_time: Optional[datetime] = None
        self._rows: list[SessionEvent] = []

        self._write_rows()

    def log_event(
        self,
        person_name: str,
        result: str,
        confidence: float,
        blink_detected: bool,
        motion_detected: bool,
    ) -> bool:
        """Record an event if it is new or sufficiently separated from the last identical event."""
        now = datetime.now(timezone.utc)
        normalized_name = person_name or "Unknown"
        normalized_result = result.upper().strip()
        rounded_confidence = round(float(confidence), 3)
        event_key = (
            normalized_name,
            normalized_result,
            rounded_confidence,
            bool(blink_detected),
            bool(motion_detected),
        )

        if self._last_event_key == event_key and self._last_event_time is not None:
            elapsed = (now - self._last_event_time).total_seconds()
            if elapsed < 2.0:
                return False

        self._last_event_key = event_key
        self._last_event_time = now

        event = SessionEvent(
            timestamp=now.isoformat(),
            person_name=normalized_name,
            result=normalized_result,
            confidence=rounded_confidence,
            blink_detected=bool(blink_detected),
            motion_detected=bool(motion_detected),
        )
        self._rows.append(event)
        self._write_rows()
        return True

    def _write_rows(self) -> None:
        """Persist the current session rows to disk."""
        frame = pd.DataFrame(
            [
                {
                    "Timestamp": row.timestamp,
                    "Person Name": row.person_name,
                    "Result": row.result,
                    "Confidence": row.confidence,
                    "Blink Detected": row.blink_detected,
                    "Motion Detected": row.motion_detected,
                }
                for row in self._rows
            ]
        )
        frame.to_csv(self.log_path, index=False)
