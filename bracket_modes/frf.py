"""Windows and the averaged acceleration-per-force spectrum."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bracket_modes.analyze_settings import AnalysisSettings
from bracket_modes.detect import Hit
from bracket_modes.io import Recording


@dataclass
class Spectrum:
    frequency_hz: np.ndarray
    df_hz: float
    n_fft: int
    H: dict[str, np.ndarray]
    coherence: dict[str, np.ndarray] | None
    hit_db: dict[str, list[np.ndarray]]
    excited: np.ndarray
    tau_s: float | None
    search_fmin_hz: float
    search_fmax_hz: float
    n_averaged: int
    axes: list[str] = field(default_factory=list)


def compute_spectrum(
    recording: Recording,
    hits: list[Hit],
    settings: AnalysisSettings,
    live_axes: list[str],
) -> Spectrum | None:
    accepted = [hit for hit in hits if hit.accepted]
    if not accepted or not live_axes:
        return None

    fs = recording.sample_rate_hz
    lengths = [hit.end_index - hit.start_index for hit in accepted]
    n_fft = int(min(lengths))
    if n_fft < 32:
        return None

    frequency = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    df = float(frequency[1] - frequency[0]) if frequency.size > 1 else fs / n_fft
    fmax = settings.fmax_hz if settings.fmax_hz is not None else 0.4 * fs
    fmax = float(min(fmax, frequency[-1]))
    fmin = float(settings.fmin_hz)

    gff = np.zeros(frequency.shape, dtype=np.complex128)
    gaf = {axis: np.zeros(frequency.shape, dtype=np.complex128) for axis in live_axes}
    gaa = {axis: np.zeros(frequency.shape, dtype=np.complex128) for axis in live_axes}
    hit_db = {axis: [] for axis in live_axes}
    tau_s: float | None = None

    for hit in accepted:
        start = hit.start_index
        force = np.asarray(recording.force[start : start + n_fft], dtype=float).copy()
        peak_offset = hit.peak_index - start
        _remove_quiet_mean(force, peak_offset)
        force_w, response_w, tau_s = make_windows(
            force, peak_offset, fs, settings.exponential_window
        )
        force_spec = np.fft.rfft(force * force_w)
        gff += force_spec * np.conj(force_spec)
        mag_f = np.abs(force_spec)
        floor = float(np.max(mag_f)) * 10 ** (-settings.excited_drop_db / 20.0)
        for axis in live_axes:
            accel = np.asarray(recording.accel[axis][start : start + n_fft], dtype=float).copy()
            _remove_quiet_mean(accel, peak_offset)
            accel_spec = np.fft.rfft(accel * response_w)
            gaf[axis] += accel_spec * np.conj(force_spec)
            gaa[axis] += accel_spec * np.conj(accel_spec)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = accel_spec / force_spec
            decibels = 20.0 * np.log10(np.maximum(np.abs(ratio), 1e-30))
            decibels[mag_f < floor] = np.nan
            hit_db[axis].append(decibels)

    n_avg = len(accepted)
    gff /= n_avg
    power = np.real(gff)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    peak_db = float(np.max(power_db[1:])) if power_db.size > 1 else float(power_db[0])
    excited = power_db >= peak_db - settings.excited_drop_db
    excited[0] = False

    H: dict[str, np.ndarray] = {}
    coherence: dict[str, np.ndarray] | None = {} if n_avg >= 2 else None
    for axis in live_axes:
        gaf[axis] /= n_avg
        gaa[axis] /= n_avg
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = gaf[axis] / gff
        ratio[~np.isfinite(ratio)] = 0
        H[axis] = ratio
        if coherence is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma = (np.abs(gaf[axis]) ** 2) / (np.real(gaa[axis]) * power)
            gamma = np.real(gamma)
            gamma[~np.isfinite(gamma)] = 0.0
            coherence[axis] = np.clip(gamma, 0.0, 1.0)

    return Spectrum(
        frequency_hz=frequency,
        df_hz=df,
        n_fft=n_fft,
        H=H,
        coherence=coherence,
        hit_db=hit_db,
        excited=excited,
        tau_s=tau_s,
        search_fmin_hz=fmin,
        search_fmax_hz=fmax,
        n_averaged=n_avg,
        axes=list(live_axes),
    )


def make_windows(
    force: np.ndarray,
    peak_offset: int,
    fs: float,
    exponential: bool,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Force window over the blow, and an optional exponential window on the response.

    The exponential window reaches 5% at the end of the block. `tau` is the time
    constant of that decay, used later to remove the damping the window added.
    """
    n = force.size
    peak_offset = int(np.clip(peak_offset, 0, n - 1))
    force_w = _force_window(force, peak_offset, fs)
    response_w = np.ones(n)
    tau: float | None = None
    if exponential and peak_offset < n - 1:
        t_after = (n - 1 - peak_offset) / fs
        tau = float(-t_after / np.log(0.05))
        dt = (np.arange(n) - peak_offset) / fs
        after = dt > 0
        response_w[after] = np.exp(-dt[after] / tau)
    taper_n = max(2, int(round(0.02 * n)))
    # Cosine taper, 1 at the start of the taper and 0 at the last sample.
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper_n)))
    response_w[-taper_n:] *= ramp
    return force_w, response_w, tau


def _force_window(force: np.ndarray, peak_offset: int, fs: float) -> np.ndarray:
    n = force.size
    window = np.zeros(n)
    peak_val = max(abs(float(force[peak_offset])), 1e-12)
    level = 0.10 * peak_val
    search_end = min(n, peak_offset + int(round(0.05 * fs)) + 1)
    above = np.flatnonzero(np.abs(force[peak_offset:search_end]) > level)
    min_after = max(1, int(round(0.002 * fs)))
    margin = max(1, int(round(0.003 * fs)))
    if above.size:
        last = peak_offset + int(above[-1])
    else:
        last = peak_offset
    last = min(n - 1, max(last, peak_offset + min_after) + margin)
    taper = max(2, int(round(0.002 * fs)))
    window[: last + 1] = 1.0
    taper_end = min(n, last + 1 + taper)
    if taper_end > last + 1:
        count = taper_end - (last + 1)
        window[last + 1 : taper_end] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, count + 2)[1:-1]))
    return window


def _remove_quiet_mean(data: np.ndarray, peak_offset: int) -> None:
    quiet_end = max(1, peak_offset // 2)
    if peak_offset >= 4:
        data -= float(np.mean(data[:quiet_end]))
