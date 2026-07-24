"""MEt3R metrics.

The legacy :class:`MEt3R` implementation has heavyweight optional dependencies.
It is imported lazily so the known-camera metric can be used independently.
"""

from __future__ import annotations

from typing import Any

from .known_camera import (
    FactorialMetricResult,
    KnownCameraConfig,
    PairMetricResult,
    SymmetricPairMetricResult,
    score_factorial_known_camera_pair,
    score_known_camera_pair,
    score_symmetric_known_camera_pair,
)

__all__ = [
    "KnownCameraConfig",
    "FactorialMetricResult",
    "MEt3R",
    "PairMetricResult",
    "SymmetricPairMetricResult",
    "score_factorial_known_camera_pair",
    "score_known_camera_pair",
    "score_symmetric_known_camera_pair",
]


def __getattr__(name: str) -> Any:
    if name == "MEt3R":
        from .met3r import MEt3R

        return MEt3R
    raise AttributeError(name)
