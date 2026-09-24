"""Run hit detection, the averaged spectrum, and peak picking as one step."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bracket_modes.analyze_settings import AnalysisSettings, settings_error
from bracket_modes.detect import Detection, Hit, channel_fault, detect_hits
from bracket_modes.frf import Spectrum, compute_spectrum
from bracket_modes.io import Recording
from bracket_modes.peaks import ModeResult, pick_modes

__all__ = ["Analysis", "AnalysisSettings", "analyze_recording", "format_hz", "settings_error"]


@dataclass
class Analysis:
    recording_name: str
    sample_rate_hz: float
    duration_s: float
    hits: list[Hit]
    threshold: float
    noise_sigma: float
    modes: list[ModeResult]
    warnings: list[str]
    notes: list[str]
    settings: AnalysisSettings
    frequency_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    df_hz: float = 0.0
    n_fft: int = 0
    H: dict[str, np.ndarray] = field(default_factory=dict)
    coherence: dict[str, np.ndarray] | None = None
    hit_db: dict[str, list[np.ndarray]] = field(default_factory=dict)
    excited: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    tau_s: float | None = None
    search_fmin_hz: float = 10.0
    search_fmax_hz: float = 0.0
    fmax_is_automatic: bool = True
    n_averaged: int = 0
    live_axes: list[str] = field(default_factory=list)

    @property
    def accepted_hits(self) -> list[Hit]:
        return [hit for hit in self.hits if hit.accepted]

    def strongest_hit(self) -> Hit | None:
        accepted = self.accepted_hits
        if not accepted:
            return None
        return max(accepted, key=lambda hit: abs(hit.peak_force))


def analyze_recording(recording: Recording, settings: AnalysisSettings | None = None) -> Analysis:
    settings = settings or AnalysisSettings()
    problem = settings_error(settings)
    if problem:
        raise ValueError(problem)

    warnings = list(recording.notes)
    live_axes: list[str] = []
    for name, data in recording.accel.items():
        fault = channel_fault(data)
        if fault == "dead":
            warnings.append(
                f"The {name.upper()} acceleration is flat, so that channel was left out."
            )
            continue
        live_axes.append(name)
        if fault == "clipped":
            warnings.append(
                f"The {name.upper()} acceleration flattens at its extreme. The sensor or the input range was probably overloaded."
            )
    force_fault = channel_fault(recording.force)
    if force_fault == "clipped":
        warnings.append(
            "The hammer signal flattens at its extreme. The hammer channel was probably overloaded."
        )

    detection: Detection = detect_hits(recording, settings)
    warnings.extend(detection.warnings)
    for hit in detection.hits:
        if hit.accepted:
            continue
        if hit.reason == "double-hit" and hit.second_peak_time_s is not None:
            warnings.append(
                f"The blow at {hit.peak_time_s:.3f} s was left out because a second blow landed at {hit.second_peak_time_s:.3f} s, while the hammer was still in contact."
            )
        elif hit.reason:
            warnings.append(f"The blow at {hit.peak_time_s:.3f} s was left out ({hit.reason}).")

    spectrum = compute_spectrum(recording, detection.hits, settings, live_axes)
    modes: list[ModeResult] = []
    notes = _processing_notes(recording, settings, detection, spectrum)
    if spectrum is None:
        if not any("No hammer hits" in warning for warning in warnings):
            warnings.append("There was not enough of a ring-down after the hits to build a spectrum.")
        spectrum_bits = _empty_spectrum_fields(settings, recording.sample_rate_hz)
    else:
        modes = pick_modes(spectrum, settings)
        _add_spectrum_warnings(warnings, spectrum)
        spectrum_bits = spectrum

    n_accepted = sum(hit.accepted for hit in detection.hits)
    if 0 < n_accepted < 3:
        warnings.append(
            f"Only {n_accepted} blow{' was' if n_accepted == 1 else 's were'} usable. Three or more make the repeatability check meaningful."
        )
    if any(hit.shortened for hit in detection.hits if hit.accepted):
        warnings.append(
            "Some blows were close together, so those records were shortened and the frequency resolution got worse."
        )

    analysis = Analysis(
        recording_name=recording.source_name,
        sample_rate_hz=recording.sample_rate_hz,
        duration_s=recording.duration_s,
        hits=detection.hits,
        threshold=detection.threshold,
        noise_sigma=detection.noise_sigma,
        modes=modes,
        warnings=_unique(warnings),
        notes=notes,
        settings=settings,
        live_axes=live_axes,
        fmax_is_automatic=settings.fmax_is_automatic,
    )
    if isinstance(spectrum_bits, Spectrum):
        analysis.frequency_hz = spectrum_bits.frequency_hz
        analysis.df_hz = spectrum_bits.df_hz
        analysis.n_fft = spectrum_bits.n_fft
        analysis.H = spectrum_bits.H
        analysis.coherence = spectrum_bits.coherence
        analysis.hit_db = spectrum_bits.hit_db
        analysis.excited = spectrum_bits.excited
        analysis.tau_s = spectrum_bits.tau_s
        analysis.search_fmin_hz = spectrum_bits.search_fmin_hz
        analysis.search_fmax_hz = spectrum_bits.search_fmax_hz
        analysis.n_averaged = spectrum_bits.n_averaged
    else:
        analysis.search_fmin_hz = spectrum_bits[0]
        analysis.search_fmax_hz = spectrum_bits[1]
    return analysis


def format_hz(frequency: float) -> str:
    if frequency >= 100:
        return f"{frequency:.0f} Hz"
    return f"{frequency:.1f} Hz"


def summary_sentence(modes: list[ModeResult]) -> str:
    if not modes:
        return "No natural frequencies were found in this recording."
    labels = [format_hz(mode.frequency_hz).removesuffix(" Hz") for mode in modes]
    if len(labels) == 1:
        listed = labels[0]
    else:
        listed = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    noun = "natural frequency" if len(labels) == 1 else "natural frequencies"
    return f"Found {len(labels)} {noun}: {listed} Hz."


def _processing_notes(recording, settings, detection, spectrum: Spectrum | None) -> list[str]:
    fs = recording.sample_rate_hz
    fmax = settings.fmax_hz if settings.fmax_hz is not None else 0.4 * fs
    notes = [
        f"Sample rate {fs:.1f} Hz, recording length {recording.duration_s:.2f} s.",
        (
            f"Each blow keeps {settings.pretrigger_s * 1000:.0f} ms before the hit and "
            f"{settings.block_length_s:.2f} s after it."
        ),
    ]
    if spectrum is not None:
        notes.append(
            f"Frequency resolution is {spectrum.df_hz:.2f} Hz. "
            f"The search ran from {spectrum.search_fmin_hz:.0f} Hz to {spectrum.search_fmax_hz:.0f} Hz."
        )
        notes.append(
            f"Averaged {spectrum.n_averaged} blows with the H1 estimator "
            "(the cross-spectrum of acceleration and force, divided by the force spectrum)."
        )
        if spectrum.tau_s is not None:
            notes.append(
                "An exponential window was applied to the acceleration so the ring-down fades "
                f"to 5% by the end of each record (time constant {spectrum.tau_s:.3f} s). "
                "The damping estimate has that extra fade removed."
            )
        else:
            notes.append("No exponential window was applied to the acceleration.")
    else:
        notes.append(f"The requested search was {settings.fmin_hz:.0f} Hz to {fmax:.0f} Hz.")
    notes.append(
        "A natural frequency here is a peak in acceleration per unit of hammer force. "
        "For a lightly damped metal bracket that peak is the natural frequency. "
        "This is one sensor location, not a mode-shape survey."
    )
    if detection.threshold > 0:
        unit = f" {recording.force_unit}" if recording.force_unit else ""
        sentence = (
            f"A blow had to reach {detection.threshold:.3g}{unit} "
            f"({settings.threshold_fraction:.0%} of the largest force)."
        )
        if detection.noise_sigma > 0:
            sentence += f" It also had to clear the force noise ({detection.noise_sigma:.3g}{unit})."
        notes.append(sentence)
    return notes


def _add_spectrum_warnings(warnings: list[str], spectrum: Spectrum) -> None:
    if spectrum.df_hz > 2.0:
        warnings.append(
            f"Frequency resolution is {spectrum.df_hz:.1f} Hz because the record after each blow is short. "
            "Frequencies closer than that will blur together."
        )
    band = (spectrum.frequency_hz >= spectrum.search_fmin_hz) & (
        spectrum.frequency_hz <= spectrum.search_fmax_hz
    )
    if np.count_nonzero(band) == 0:
        return
    excited = spectrum.excited & band
    missing = 1.0 - np.count_nonzero(excited) / np.count_nonzero(band)
    if missing > 0.5 and np.any(excited):
        rolloff = float(spectrum.frequency_hz[excited].max())
        warnings.append(
            f"The hammer's force gets weak above about {rolloff:.0f} Hz, so higher frequencies "
            "in the search range were not used. A harder hammer tip reaches higher frequencies."
        )


def _empty_spectrum_fields(settings: AnalysisSettings, fs: float) -> tuple[float, float]:
    fmax = settings.fmax_hz if settings.fmax_hz is not None else 0.4 * fs
    return float(settings.fmin_hz), float(fmax)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
