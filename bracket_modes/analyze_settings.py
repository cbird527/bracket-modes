"""Shared analysis settings, kept separate so the other modules can import them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisSettings:
    threshold_fraction: float = 0.30
    pretrigger_s: float = 0.020
    block_length_s: float = 1.0
    fmin_hz: float = 10.0
    fmax_hz: float | None = None
    coherence_min: float = 0.85
    exponential_window: bool = True
    prominence_db: float = 8.0
    max_modes: int = 10
    excited_drop_db: float = 30.0

    @property
    def fmax_is_automatic(self) -> bool:
        return self.fmax_hz is None


def settings_error(settings: AnalysisSettings) -> str | None:
    if not 0.01 <= settings.threshold_fraction <= 0.99:
        return "The hit threshold must be between 1% and 99% of the largest blow."
    if not 0.0 <= settings.pretrigger_s <= 2.0:
        return "The time before the hit must be between 0 and 2000 ms."
    if not 0.05 <= settings.block_length_s <= 30.0:
        return "The time kept after the hit must be between 0.05 and 30 seconds."
    if settings.fmin_hz < 0:
        return "The lower frequency cannot be negative."
    if settings.fmax_hz is not None and settings.fmax_hz <= settings.fmin_hz:
        return "The upper frequency must be higher than the lower frequency."
    if not 0.0 <= settings.coherence_min <= 1.0:
        return "Repeatability must be between 0 and 1."
    return None
