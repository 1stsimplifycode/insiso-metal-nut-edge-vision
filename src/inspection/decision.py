"""Inspection decision policy, kept separate from the detector and from I/O."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionPolicy:
    reject_conf: float          # any detection >= this -> REJECT (validation-selected operating point)
    review_conf: float | None   # max score in [review_conf, reject_conf) -> REVIEW (human check)

    def decide(self, det: np.ndarray) -> str:
        top = float(det[:, 4].max()) if len(det) else 0.0
        if top >= self.reject_conf:
            return "REJECT"
        if self.review_conf is not None and top >= self.review_conf:
            return "REVIEW"
        return "PASS"
