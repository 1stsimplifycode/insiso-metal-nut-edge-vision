"""SIMULATED controller interface: serialises inspection results as structured JSON messages.

Not tested on any PLC / Bosch Rexroth ctrlX hardware. A real integration would replace the
sink (e.g. OPC UA variable writes, MQTT publish, or the ctrlX Data Layer) and add a
handshake/acknowledgement, a watchdog and a fail-safe default (REJECT on timeout/error).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO


@dataclass
class InspectionMessage:
    sample_id: str
    decision: str                       # PASS | REJECT | REVIEW | ERROR
    detections: list[dict]              # [{"box_xyxy": [...], "confidence": float, "class": str}]
    max_confidence: float
    processing_ms: float                # capture->decision, measured in the app
    model_version: str
    input_resolution: str
    timestamp_utc: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"))
    schema: str = "insiso-demo/inspection-result/v1"


class JsonlSink:
    """Writes one JSON object per line to a file and/or a stream."""

    def __init__(self, path: Path | None = None, stream: TextIO | None = sys.stdout):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8") if path else None
        self.stream = stream

    def send(self, msg: InspectionMessage) -> None:
        line = json.dumps(asdict(msg), separators=(",", ":"))
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()
        if self.stream:
            print(line, file=self.stream)

    def close(self) -> None:
        if self.f:
            self.f.close()
