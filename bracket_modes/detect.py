"""Find hammer hits, double-hits, and clipped channels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from bracket_modes.analyze_settings import AnalysisSettings
from bracket_modes.io import Recording

# A second blow inside this window is a double-hit, not a new average.
_DOUBLE_MIN_S = 0.025
_DOUBLE_MAX_S = 0.050
_MIN_PEAK_GAP_S = 0.005
_MIN_BLOCK_SAMPLES = 32


@dataclass
class Hit:
    peak_index: int
    peak_time_s: float
    peak_force: float
    accepted: bool
    reason: str
    start_index: int
    end_index: int
    shortened: bool = False
    second_peak_time_s: float | None = None


@dataclass
class Detection:
    hits: list[Hit] = field(default_factory=list)
    threshold: float = 0.0
    noise_sigma: float = 0.0
    warnings: list[str] = field(default_factory=list)


def detect_hits(recording: Recording, settings: AnalysisSettings) -> Detection:
    force = recording.force
    fs = recording.sample_rate_hz
    n = force.size
    sigma = _robust_sigma(force)
    largest = float(np.max(np.abs(force))) if n else 0.0
    from_fraction = settings.threshold_fraction * largest
    from_noise = 8.0 * sigma
    threshold = float(max(from_fraction, from_noise))

    detection = Detection(threshold=threshold, noise_sigma=float(sigma))
    if n < _MIN_BLOCK_SAMPLES or largest <= 0:
        detection.warnings.append(
            "No hammer hits were found. Check the force column, or lower the hit threshold."
        )
        return detection

    distance = max(1, int(round(_MIN_PEAK_GAP_S * fs)))
    peaks, _props = find_peaks(np.abs(force), height=threshold, distance=distance)
    if peaks.size == 0:
        detection.warnings.append(
            "No hammer hits were found above the threshold. Lower the hit threshold or check the force column."
        )
        return detection

    pre = int(round(settings.pretrigger_s * fs))
    block = int(round(settings.block_length_s * fs))
    raw = [int(p) for p in peaks]
    used: set[int] = set()
    hits: list[Hit] = []
    for i, peak in enumerate(raw):
        if i in used:
            continue
        gate = _force_activity_s(force, peak, fs)
        gate_end = peak + max(1, int(round(gate * fs)))
        group = [i]
        for j in range(i + 1, len(raw)):
            if raw[j] <= gate_end:
                group.append(j)
            else:
                break
        for j in group:
            used.add(j)
        second = recording.time_s[raw[group[1]]] if len(group) > 1 else None
        hits.append(
            Hit(
                peak_index=peak,
                peak_time_s=float(recording.time_s[peak]),
                peak_force=float(force[peak]),
                accepted=len(group) == 1,
                reason="" if len(group) == 1 else "double-hit",
                start_index=peak,
                end_index=peak,
                second_peak_time_s=None if second is None else float(second),
            )
        )

    for i, hit in enumerate(hits):
        start = max(0, hit.peak_index - pre)
        end = min(n, hit.peak_index + block)
        if i + 1 < len(hits):
            end = min(end, max(hit.peak_index + 1, hits[i + 1].peak_index - pre))
        nominal_end = min(n, hit.peak_index + block)
        hit.start_index = start
        hit.end_index = end
        if not hit.accepted:
            continue
        if end - start < _MIN_BLOCK_SAMPLES or end <= hit.peak_index:
            hit.accepted = False
            hit.reason = "too close to another hit"
            continue
        hit.shortened = end < nominal_end

    detection.hits = hits
    n_accepted = sum(hit.accepted for hit in hits)
    if n_accepted == 0 and not detection.warnings:
        detection.warnings.append(
            "Every hammer blow was left out. Look at the recording tab for double-hits or blows that are too close together."
        )
    return detection


def channel_fault(data: np.ndarray) -> str | None:
    """Return 'dead', 'clipped', or None.

    Clipping is a run of identical samples sitting on the extreme value, well
    away from the middle of the signal. A single sharp peak is not clipping.
    """
    if data.size < 8 or not np.isfinite(data).all():
        return None
    peak_to_peak = float(np.ptp(data))
    if peak_to_peak == 0.0:
        return "dead"
    max_v = float(np.max(data))
    min_v = float(np.min(data))
    longest = max(_longest_equal_run(data, max_v), _longest_equal_run(data, min_v))
    center = float(np.median(data))
    scale = _robust_sigma(data)
    extreme = max(abs(max_v - center), abs(min_v - center))
    if longest >= 5 and extreme > max(20.0 * scale, 0.05 * peak_to_peak):
        return "clipped"
    return None


def _force_activity_s(force: np.ndarray, peak: int, fs: float) -> float:
    peak_val = abs(float(force[peak]))
    if peak_val <= 0:
        return _DOUBLE_MIN_S
    level = 0.10 * peak_val
    search_end = min(force.size, peak + int(round(_DOUBLE_MAX_S * fs)) + 1)
    above = np.flatnonzero(np.abs(force[peak:search_end]) > level)
    if above.size == 0:
        duration = 1.0 / fs
    else:
        duration = float(above[-1]) / fs
    return float(min(_DOUBLE_MAX_S, max(_DOUBLE_MIN_S, duration + 0.003)))


def _robust_sigma(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    center = float(np.median(data))
    return float(1.4826 * np.median(np.abs(data - center)))


def _longest_equal_run(data: np.ndarray, value: float) -> int:
    mask = np.asarray(data == value, dtype=np.int8)
    if mask.size == 0 or not np.any(mask):
        return 0
    padded = np.concatenate(([0], mask, [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if starts.size == 0:
        return 0
    return int(np.max(ends - starts))
