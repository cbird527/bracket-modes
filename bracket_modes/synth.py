"""Write a realistic 10-hit impulse recording with known natural frequencies.

The response is the exact sampled accelerance of four lightly damped modes, so the
planted frequencies are not an approximation of a time-stepper. Measurement noise,
a weak 60 Hz hum, and hit-to-hit scatter are added after that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FS = 10240.0
SEED = 7
MODES = (
    {"hz": 103.0, "zeta": 0.009, "R": (0.40, 0.03, 0.14)},
    {"hz": 268.0, "zeta": 0.007, "R": (0.02, 0.12, 0.02)},
    {"hz": 511.0, "zeta": 0.010, "R": (0.04, 0.005, 0.035)},
    {"hz": 874.0, "zeta": 0.008, "R": (0.008, 0.010, 0.040)},
)
HEADER = "time_s,force_N,accel_x_g,accel_y_g,accel_z_g"


def example_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples"


def generate(seed: int = SEED) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    hit_times = [0.40]
    for _ in range(9):
        hit_times.append(hit_times[-1] + float(rng.uniform(2.2, 3.2)))
    widths = rng.uniform(0.0006, 0.0009, size=10)
    amplitudes = rng.uniform(50.0, 150.0, size=10)
    mode_scale = rng.uniform(0.92, 1.08, size=(10, len(MODES)))
    axis_jitter = rng.uniform(0.97, 1.03, size=(10, len(MODES), 3))

    t_end = hit_times[-1] + 1.8
    n = int(np.ceil(t_end * FS)) + 1
    time_s = np.arange(n) / FS
    force = np.zeros(n)
    accel = [np.zeros(n), np.zeros(n), np.zeros(n)]
    pre = int(round(0.05 * FS))
    nseg = int(round(1.60 * FS))

    for hit_index, t_peak in enumerate(hit_times):
        _add_half_sine(force, t_peak, float(widths[hit_index]), float(amplitudes[hit_index]))
        peak_index = int(round(t_peak * FS))
        i0 = peak_index - pre
        if i0 < 0 or i0 + nseg > n:
            raise RuntimeError("The generated recording is too short for a hit segment.")
        segment = force[i0 : i0 + nseg]
        for mode_index, mode in enumerate(MODES):
            residues = []
            for axis in range(3):
                gain = mode["R"][axis] * mode_scale[hit_index, mode_index] * axis_jitter[hit_index, mode_index, axis]
                residues.append(float(gain))
            _add_mode(accel, segment, i0, mode["hz"], mode["zeta"], residues)

    force += rng.normal(0.0, 0.15, n)
    force += 0.02
    hum_phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
    for axis in range(3):
        accel[axis] += 0.005
        accel[axis] += 0.01 * np.sin(2.0 * np.pi * 60.0 * time_s + hum_phase[axis])
        accel[axis] += rng.normal(0.0, 0.02, n)

    data = np.column_stack([time_s, force, accel[0], accel[1], accel[2]])
    truth = {
        "seed": seed,
        "sample_rate_hz": FS,
        "hit_count": 10,
        "hit_times_s": [round(t, 6) for t in hit_times],
        "modes_hz": [mode["hz"] for mode in MODES],
        "hum_hz": 60.0,
        "n_rows": int(n),
    }
    return data, truth


def write_example(directory: Path | None = None) -> Path:
    directory = example_dir() if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data, truth = generate()
    csv_path = directory / "bracket_10hits.csv"
    truth_path = directory / "bracket_10hits_truth.json"
    # Eight decimals keeps the sample spacing even after the file is round-tripped.
    np.savetxt(csv_path, data, delimiter=",", header=HEADER, comments="", fmt="%.8f")
    truth_path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    return csv_path


def _add_half_sine(force: np.ndarray, t_peak: float, width: float, amplitude: float) -> None:
    t0 = t_peak - width / 2.0
    t1 = t_peak + width / 2.0
    i0 = max(0, int(np.floor(t0 * FS)))
    i1 = min(force.size - 1, int(np.ceil(t1 * FS)))
    idx = np.arange(i0, i1 + 1)
    phase = (idx / FS - t0) / width
    keep = (phase >= 0.0) & (phase <= 1.0)
    force[idx[keep]] += amplitude * np.sin(np.pi * phase[keep])


def _add_mode(
    accel: list[np.ndarray],
    segment: np.ndarray,
    i0: int,
    freq_hz: float,
    zeta: float,
    residues: list[float],
) -> None:
    nseg = segment.size
    freq = np.fft.rfftfreq(nseg, d=1.0 / FS)
    s = 2j * np.pi * freq
    omega = 2.0 * np.pi * freq_hz
    denom = s * s + 2.0 * zeta * omega * s + omega * omega
    transfer = np.zeros(s.shape, dtype=np.complex128)
    nonzero = denom != 0
    transfer[nonzero] = (s[nonzero] ** 2) / denom[nonzero]
    response = np.fft.irfft(np.fft.rfft(segment) * transfer, n=nseg)
    for axis, gain in enumerate(residues):
        accel[axis][i0 : i0 + nseg] += gain * response


if __name__ == "__main__":
    path = write_example()
    truth = json.loads((path.parent / "bracket_10hits_truth.json").read_text(encoding="utf-8"))
    print(f"Wrote {path} ({truth['n_rows']} rows)")
    print(f"Hits at {', '.join(f'{t:.3f} s' for t in truth['hit_times_s'])}")
    print(f"Planted modes: {', '.join(str(f) for f in truth['modes_hz'])} Hz")
