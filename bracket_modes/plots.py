"""Charts shared by the window and the HTML report."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from bracket_modes.analyze import Analysis, format_hz
from bracket_modes.frf import make_windows
from bracket_modes.io import Recording

FORCE_COLOR = "#333333"
AXIS_COLOR = {"x": "#0072B2", "y": "#E69F00", "z": "#009E73"}
REJECT_COLOR = "#D55E00"
COMBINED_COLOR = "#555555"
AXIS_ORDER = ("x", "y", "z")


def draw_recording(fig: Figure, recording: Recording, analysis: Analysis) -> None:
    fig.clear()
    fig.set_facecolor("white")
    channels = [_force_channel(recording)]
    for axis in AXIS_ORDER:
        if axis in recording.accel:
            channels.append(_accel_channel(recording, axis))
    axes = fig.subplots(len(channels), 1, sharex=True, squeeze=False)[:, 0]
    labeled_used = False
    labeled_left_out = False
    for ax, (label, series, color) in zip(axes, channels):
        shown_t, shown_y = _envelope(recording.time_s, series)
        ax.plot(shown_t, shown_y, color=color, lw=0.8)
        ax.set_ylabel(label, fontsize=8)
        _style(ax)
        for hit in analysis.hits:
            if hit.accepted and not labeled_used and ax is axes[0]:
                legend = "Blow used"
                labeled_used = True
            elif not hit.accepted and not labeled_left_out and ax is axes[0]:
                legend = "Blow left out"
                labeled_left_out = True
            else:
                legend = None
            ax.axvline(
                hit.peak_time_s,
                color=FORCE_COLOR if hit.accepted else REJECT_COLOR,
                lw=0.7,
                alpha=0.75,
                label=legend,
            )
    if analysis.threshold > 0:
        axes[0].axhline(
            analysis.threshold,
            color="#888888",
            ls="--",
            lw=0.8,
            label="Hit threshold",
        )
    if len(analysis.hits) <= 40:
        # Numbers sit along the top so a short blow is labeled as clearly as a hard one.
        number_coords = blended_transform_factory(axes[0].transData, axes[0].transAxes)
        for number, hit in enumerate(analysis.hits, start=1):
            axes[0].annotate(
                str(number),
                (hit.peak_time_s, 0.97),
                xycoords=number_coords,
                ha="center",
                va="top",
                fontsize=8,
                color=FORCE_COLOR if hit.accepted else REJECT_COLOR,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.9},
            )
    axes[0].set_title("Hammer hits found in the recording")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)
    _finish(fig)


def draw_hit(fig: Figure, recording: Recording, analysis: Analysis) -> None:
    fig.clear()
    fig.set_facecolor("white")
    hit = analysis.strongest_hit()
    if hit is None or analysis.n_fft < 32:
        _message(fig, "No blow was accepted, so there is nothing to window.")
        return
    start = hit.start_index
    n = min(analysis.n_fft, recording.force.size - start)
    peak_offset = hit.peak_index - start
    if n < 32 or not 0 <= peak_offset < n:
        _message(fig, "The accepted blow is too short to show.")
        return

    force = np.asarray(recording.force[start : start + n], dtype=float)
    time_s = recording.time_s[start : start + n]
    force_w, response_w, _tau = make_windows(
        force, peak_offset, recording.sample_rate_hz, analysis.settings.exponential_window
    )
    channels = [("Hammer force" + _unit(recording.force_unit), force, force_w, FORCE_COLOR)]
    for axis in AXIS_ORDER:
        if axis not in recording.accel:
            continue
        series = np.asarray(recording.accel[axis][start : start + n], dtype=float)
        label = f"{axis.upper()} acceleration" + _unit(recording.accel_unit.get(axis, ""))
        channels.append((label, series, response_w, AXIS_COLOR[axis]))

    axes = fig.subplots(len(channels), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (label, series, window, color) in zip(axes, channels):
        ax.plot(time_s, series, color=color, lw=0.9)
        scale = float(np.max(np.abs(series))) or 1.0
        ax.fill_between(
            time_s,
            0.0,
            window * scale,
            color=color,
            alpha=0.18 if ax is axes[0] else 0.07,
        )
        ax.set_ylabel(label, fontsize=8)
        _style(ax)
    axes[0].set_title(
        "One blow, and the parts of the signal that are used\n"
        "Shading is the part included. The box zooms in on the hammer pulse."
    )
    axes[-1].set_xlabel("Time (s)")
    _add_pulse_inset(axes[0], time_s, force, force_w, peak_offset, recording.sample_rate_hz)
    _finish(fig)


def draw_frf(fig: Figure, recording: Recording, analysis: Analysis, show_combined: bool = True) -> None:
    fig.clear()
    fig.set_facecolor("white")
    ax = fig.add_subplot(111)
    if analysis.frequency_hz.size == 0 or not analysis.H:
        _message(fig, "No frequency response yet.", title="How strongly the bracket responded")
        return

    fmin, fright, caption = view_limits(analysis)
    freq = analysis.frequency_hz
    visible = (freq >= analysis.search_fmin_hz) & (freq <= analysis.search_fmax_hz) & analysis.excited
    for axis in AXIS_ORDER:
        if axis not in analysis.H:
            continue
        color = AXIS_COLOR[axis]
        for trace in analysis.hit_db.get(axis, []):
            ax.plot(freq, trace, color=color, lw=0.5, alpha=0.18)
        decibels = 20.0 * np.log10(np.maximum(np.abs(analysis.H[axis]), 1e-30))
        decibels = decibels.astype(float)
        decibels[~visible] = np.nan
        ax.plot(freq, decibels, color=color, lw=1.8, label=f"{axis.upper()} average")

    if show_combined and len(analysis.H) >= 2:
        stacked = np.vstack([np.abs(analysis.H[axis]) for axis in AXIS_ORDER if axis in analysis.H])
        combined = np.sqrt(np.sum(stacked**2, axis=0))
        decibels = 20.0 * np.log10(np.maximum(combined, 1e-30))
        decibels[~visible] = np.nan
        ax.plot(freq, decibels, color=COMBINED_COLOR, lw=1.2, ls="--", label="Combined")

    left, right = fmin, fright
    offsets: list[int] = []
    marked: list[float] = []
    for mode in analysis.modes:
        if not (left <= mode.frequency_hz <= right):
            continue
        color = AXIS_COLOR.get(mode.strongest_axis, FORCE_COLOR)
        ax.scatter([mode.frequency_hz], [mode.magnitude_db], color=color, s=28, zorder=5)
        for axis in mode.axes:
            if axis == mode.strongest_axis or axis not in analysis.H:
                continue
            ax.scatter(
                [mode.frequency_hz],
                [20.0 * np.log10(max(np.abs(analysis.H[axis][_nearest(freq, mode.frequency_hz)]), 1e-30))],
                color=AXIS_COLOR[axis],
                s=16,
                zorder=4,
            )
        offset = 8
        if marked and abs(mode.frequency_hz - marked[-1]) < 0.08 * max(right - left, 1.0):
            offset = 22 if offsets[-1] == 8 else 8
        marked.append(mode.frequency_hz)
        offsets.append(offset)
        ax.annotate(
            format_hz(mode.frequency_hz),
            (mode.frequency_hz, mode.magnitude_db),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    handles, labels = ax.get_legend_handles_labels()
    if analysis.hit_db:
        handles.append(Line2D([0], [0], color="#999999", lw=0.8, alpha=0.8))
        labels.append("Each blow")
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)
    ax.set_xlim(left, right)
    ax.set_xlabel("Frequency (Hz)" + (f"\n{caption}" if caption else ""))
    ax.set_ylabel(_magnitude_label(recording, list(analysis.H)))
    ax.set_title("How strongly the bracket responded")
    _style(ax)
    _finish(fig)


def draw_coherence(fig: Figure, analysis: Analysis) -> None:
    fig.clear()
    fig.set_facecolor("white")
    ax = fig.add_subplot(111)
    if not analysis.coherence or analysis.frequency_hz.size == 0:
        _message(
            fig,
            "Repeatability needs at least two accepted blows.",
            title="How repeatable each frequency was",
        )
        return
    fmin, fright, _caption = view_limits(analysis)
    freq = analysis.frequency_hz
    for axis in AXIS_ORDER:
        if axis not in analysis.coherence:
            continue
        ax.plot(freq, analysis.coherence[axis], color=AXIS_COLOR[axis], lw=1.3, label=axis.upper())
    ax.axhline(
        analysis.settings.coherence_min,
        color="#888888",
        ls="--",
        lw=0.9,
        label="Accept line",
    )
    ax.set_xlim(fmin, fright)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title("How repeatable each frequency was across blows (1 = the blows agreed)")
    ax.legend(loc="lower right", fontsize=8, ncol=4)
    _style(ax)
    _finish(fig)


def view_limits(analysis: Analysis) -> tuple[float, float, str | None]:
    """X limits for the frequency charts.

    A typed upper frequency is honored exactly. The automatic search still runs
    out to 40% of the sample rate, but the chart zooms to the frequencies that
    were actually found so a low mode is not pinned against the axis.
    """
    fmin = float(analysis.search_fmin_hz)
    fmax = float(analysis.search_fmax_hz or (fmin + 1.0))
    if fmax <= fmin:
        fmax = fmin + 1.0
    if not analysis.fmax_is_automatic or not analysis.modes:
        return fmin, fmax, None
    highest = max(mode.frequency_hz for mode in analysis.modes)
    right = min(fmax, max(highest * 1.3, highest + 100.0, fmin + 100.0))
    caption = None
    if fmax - right > 0.05 * max(fmax, 1.0):
        caption = (
            f"Searched up to {fmax:.0f} Hz. The chart is zoomed to the frequencies that were found."
        )
    return fmin, right, caption


def _force_channel(recording: Recording) -> tuple[str, np.ndarray, str]:
    return ("Hammer force" + _unit(recording.force_unit), recording.force, FORCE_COLOR)


def _accel_channel(recording: Recording, axis: str) -> tuple[str, np.ndarray, str]:
    unit = recording.accel_unit.get(axis, "")
    return (f"{axis.upper()} acceleration" + _unit(unit), recording.accel[axis], AXIS_COLOR[axis])


def _unit(unit: str) -> str:
    return f" ({unit})" if unit else ""


def _magnitude_label(recording: Recording, axes: list[str]) -> str:
    units = {recording.accel_unit.get(axis, "") for axis in axes}
    force = recording.force_unit
    if len(units) == 1:
        accel = units.pop()
        if accel and force:
            return f"Magnitude (dB re 1 {accel}/{force})"
    return "Magnitude (dB, acceleration / force)"


def _nearest(freq: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(freq - value)))


def _envelope(time_s: np.ndarray, values: np.ndarray, max_buckets: int = 4000):
    n = values.size
    if n <= max_buckets * 2:
        return time_s, values
    bucket = int(np.ceil(n / max_buckets))
    parts_t = []
    parts_y = []
    for start in range(0, n, bucket):
        stop = min(n, start + bucket)
        chunk_t = time_s[start:stop]
        chunk_y = values[start:stop]
        low = int(np.argmin(chunk_y))
        high = int(np.argmax(chunk_y))
        order = (low, high) if low <= high else (high, low)
        for index in order:
            parts_t.append(chunk_t[index])
            parts_y.append(chunk_y[index])
    return np.asarray(parts_t), np.asarray(parts_y)


def _add_pulse_inset(ax, time_s, force, force_w, peak_offset, fs: float) -> None:
    rel_ms = (np.arange(force.size) - peak_offset) / fs * 1000.0
    active = np.flatnonzero(force_w > 0.05)
    right_ms = 15.0
    if active.size:
        right_ms = max(15.0, float(rel_ms[active[-1]]) + 2.0)
    inset = ax.inset_axes([0.55, 0.14, 0.42, 0.48])
    keep = (rel_ms >= -2.0) & (rel_ms <= right_ms)
    if not np.any(keep):
        inset.remove()
        return
    inset.plot(rel_ms[keep], force[keep], color=FORCE_COLOR, lw=1.0)
    inset.set_xlim(-2.0, right_ms)
    inset.set_title("Hammer pulse", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.set_xlabel("ms", fontsize=7)
    inset.grid(True, color="#eeeeee", lw=0.5)


def _style(ax) -> None:
    ax.grid(True, color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def _message(fig: Figure, text: str, title: str | None = None) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    _finish(fig)


def _finish(fig: Figure) -> None:
    try:
        fig.tight_layout()
    except ValueError:
        pass
