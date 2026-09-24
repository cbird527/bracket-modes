"""Pick natural frequencies from the averaged spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from bracket_modes.analyze_settings import AnalysisSettings
from bracket_modes.frf import Spectrum

_AXIS_ORDER = ("x", "y", "z")
_PHASE_TOLERANCE_DEG = 40.0


@dataclass
class ModeResult:
    frequency_hz: float
    axes: list[str]
    strongest_axis: str
    coherence: float | None
    damping_ratio: float | None
    damping_reliable: bool
    quality: str
    prominence_db: float
    magnitude: float
    magnitude_db: float
    phase_deg: float


@dataclass
class _AxisPeak:
    axis: str
    frequency_hz: float
    bin_index: int
    magnitude: float
    magnitude_db: float
    prominence_db: float
    coherence: float | None
    phase_deg: float
    damping_ratio: float | None
    damping_reliable: bool


def pick_modes(spectrum: Spectrum, settings: AnalysisSettings) -> list[ModeResult]:
    found: list[_AxisPeak] = []
    for axis in spectrum.axes:
        found.extend(_peaks_for_axis(spectrum, axis, settings))
    if not found:
        return []

    # A couple of bins, so the same mode on two axes is not listed twice.
    tolerance = max(spectrum.df_hz * 3.0, 2.0)
    groups = _cluster(found, tolerance)
    modes: list[ModeResult] = []
    for group in groups:
        by_axis: dict[str, _AxisPeak] = {}
        for peak in group:
            current = by_axis.get(peak.axis)
            if current is None or peak.prominence_db > current.prominence_db:
                by_axis[peak.axis] = peak
        members = list(by_axis.values())
        weights = np.array([max(member.magnitude, 1e-30) for member in members])
        frequency = float(
            np.average([member.frequency_hz for member in members], weights=weights)
        )
        strongest = max(members, key=lambda member: member.magnitude)
        coherences = [member.coherence for member in members if member.coherence is not None]
        coherence = float(min(coherences)) if coherences else None
        phase_ok = _near_quadrature(strongest.phase_deg)
        coherence_ok = coherence is None or coherence >= settings.coherence_min
        modes.append(
            ModeResult(
                frequency_hz=frequency,
                axes=[axis for axis in _AXIS_ORDER if axis in by_axis],
                strongest_axis=strongest.axis,
                coherence=coherence,
                damping_ratio=strongest.damping_ratio if strongest.damping_reliable else None,
                damping_reliable=strongest.damping_reliable,
                quality="good" if phase_ok and coherence_ok else "check",
                prominence_db=float(strongest.prominence_db),
                magnitude=float(strongest.magnitude),
                magnitude_db=float(strongest.magnitude_db),
                phase_deg=float(strongest.phase_deg),
            )
        )

    modes.sort(key=lambda mode: mode.prominence_db, reverse=True)
    modes = modes[: settings.max_modes]
    modes.sort(key=lambda mode: mode.frequency_hz)
    return modes


def _peaks_for_axis(spectrum: Spectrum, axis: str, settings: AnalysisSettings) -> list[_AxisPeak]:
    freq = spectrum.frequency_hz
    ratio = spectrum.H[axis]
    magnitude = np.abs(ratio)
    mag_db = 20.0 * np.log10(np.maximum(magnitude, 1e-30))
    valid = (
        spectrum.excited
        & (freq >= spectrum.search_fmin_hz)
        & (freq <= spectrum.search_fmax_hz)
    )
    if spectrum.coherence is not None:
        valid = valid & (spectrum.coherence[axis] >= settings.coherence_min)
    if np.count_nonzero(valid) < 5:
        return []

    # Prominence is how far the peak rises above the higher of the valleys on
    # either side, so a broad skirt does not hide a real resonance.
    indices, props = find_peaks(mag_db, prominence=settings.prominence_db, distance=3)
    prominences = props.get("prominences", [])

    peaks: list[_AxisPeak] = []
    for count, index in enumerate(indices):
        index = int(index)
        if index <= 0 or index >= mag_db.size - 1 or not valid[index]:
            continue
        delta = _parabolic(mag_db[index - 1], mag_db[index], mag_db[index + 1])
        frequency = float(freq[index] + delta * spectrum.df_hz)
        if frequency < spectrum.search_fmin_hz or frequency > spectrum.search_fmax_hz:
            continue
        coherence = None
        if spectrum.coherence is not None:
            coherence = float(spectrum.coherence[axis][index])
        damping, reliable = _damping(
            freq, magnitude, index, valid, frequency, spectrum.tau_s, spectrum.df_hz
        )
        peaks.append(
            _AxisPeak(
                axis=axis,
                frequency_hz=frequency,
                bin_index=index,
                magnitude=float(magnitude[index]),
                magnitude_db=float(mag_db[index]),
                prominence_db=float(prominences[count]),
                coherence=coherence,
                phase_deg=float(np.degrees(np.angle(ratio[index]))),
                damping_ratio=damping,
                damping_reliable=reliable,
            )
        )
    return peaks


def _parabolic(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (left - right) / denom
    return float(np.clip(delta, -1.0, 1.0))


def _damping(
    freq: np.ndarray,
    magnitude: np.ndarray,
    index: int,
    valid: np.ndarray,
    peak_frequency: float,
    tau_s: float | None,
    df: float,
) -> tuple[float | None, bool]:
    target = magnitude[index] / np.sqrt(2.0)
    left = _half_power_frequency(freq, magnitude, valid, index, target, step=-1)
    right = _half_power_frequency(freq, magnitude, valid, index, target, step=1)
    if left is None or right is None or right <= left or peak_frequency <= 0:
        return None, False
    width = right - left
    if width < df:
        return None, False
    zeta = width / (2.0 * peak_frequency)
    if tau_s is not None and tau_s > 0:
        zeta -= 1.0 / (2.0 * np.pi * peak_frequency * tau_s)
    reliable = 0.0 < zeta < 0.2
    if not reliable:
        return None, False
    return float(zeta), True


def _half_power_frequency(
    freq: np.ndarray,
    magnitude: np.ndarray,
    valid: np.ndarray,
    index: int,
    target: float,
    step: int,
) -> float | None:
    i = index
    while 0 < i < magnitude.size - 1:
        nxt = i + step
        if nxt < 0 or nxt >= magnitude.size or not valid[nxt]:
            return None
        if magnitude[nxt] <= target:
            lo, hi = (nxt, i) if nxt < i else (i, nxt)
            return _interpolate(freq, magnitude, lo, hi, target)
        # Stop if the curve climbs back to the peak before crossing half power.
        if magnitude[nxt] > magnitude[index] * 1.02:
            return None
        i = nxt
    return None


def _interpolate(freq: np.ndarray, magnitude: np.ndarray, lo: int, hi: int, target: float) -> float:
    y0 = float(magnitude[lo])
    y1 = float(magnitude[hi])
    if y1 == y0:
        return float(freq[hi])
    frac = float(np.clip((target - y0) / (y1 - y0), 0.0, 1.0))
    return float(freq[lo] + frac * (freq[hi] - freq[lo]))


def _cluster(peaks: list[_AxisPeak], tolerance: float) -> list[list[_AxisPeak]]:
    ordered = sorted(peaks, key=lambda peak: peak.frequency_hz)
    groups: list[list[_AxisPeak]] = []
    for peak in ordered:
        if not groups:
            groups.append([peak])
            continue
        center = float(np.mean([item.frequency_hz for item in groups[-1]]))
        if peak.frequency_hz - center <= tolerance:
            groups[-1].append(peak)
        else:
            groups.append([peak])
    return groups


def _near_quadrature(phase_deg: float) -> bool:
    phase = (phase_deg + 180.0) % 360.0 - 180.0
    return min(abs(phase - 90.0), abs(phase + 90.0)) <= _PHASE_TOLERANCE_DEG
