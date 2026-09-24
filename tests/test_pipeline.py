"""Recovery tests for the hammer-blow natural-frequency tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bracket_modes.analyze import AnalysisSettings, analyze_recording, summary_sentence  # noqa: E402
from bracket_modes.io import (  # noqa: E402
    build_recording,
    propose_mapping,
    read_table,
    recording_from_channels,
)
from bracket_modes.report import write_report  # noqa: E402

EXAMPLE = ROOT / "examples" / "bracket_10hits.csv"
TRUTH = ROOT / "examples" / "bracket_10hits_truth.json"


class ExampleRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EXAMPLE.exists() or not TRUTH.exists():
            raise unittest.SkipTest("Run python3 -m bracket_modes.synth to create the example recording.")
        cls.truth = json.loads(TRUTH.read_text(encoding="utf-8"))
        cls.table = read_table(EXAMPLE)
        cls.mapping = propose_mapping(cls.table)
        cls.recording = build_recording(cls.table, cls.mapping)
        cls.result = analyze_recording(cls.recording, AnalysisSettings())

    def test_columns_are_recognized(self):
        self.assertTrue(self.mapping.confident)
        self.assertEqual(self.mapping.time, 0)
        self.assertEqual(self.mapping.force, 1)
        self.assertEqual((self.mapping.x, self.mapping.y, self.mapping.z), (2, 3, 4))
        self.assertEqual(self.recording.force_unit, "N")
        self.assertEqual(self.recording.accel_unit["x"], "g")
        self.assertFalse(self.recording.resampled)
        self.assertAlmostEqual(self.recording.sample_rate_hz, 10240.0, delta=1.0)

    def test_ten_hits_and_planted_frequencies(self):
        accepted = self.result.accepted_hits
        freqs = [mode.frequency_hz for mode in self.result.modes]
        detail = (
            f"frequencies={freqs}\n"
            f"hits={[(hit.peak_time_s, hit.accepted, hit.reason) for hit in self.result.hits]}\n"
            f"warnings={self.result.warnings}"
        )
        self.assertEqual(len(accepted), 10, detail)
        self.assertGreaterEqual(freqs[0], 98.0, detail)
        self.assertLessEqual(freqs[0], 108.0, detail)
        for planted in self.truth["modes_hz"]:
            limit = max(2.0, 0.01 * planted)
            self.assertTrue(any(abs(found - planted) <= limit for found in freqs), detail)
        self.assertFalse(any(55.0 <= found <= 70.0 for found in freqs), detail)
        for expected, hit in zip(self.truth["hit_times_s"], accepted):
            self.assertLess(abs(hit.peak_time_s - expected), 0.001, detail)

    def test_report_contains_the_answer_and_charts(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "report.html"
            csv_path = write_report(html_path, self.recording, self.result)
            text = html_path.read_text(encoding="utf-8")
            self.assertIn(summary_sentence(self.result.modes), text)
            self.assertEqual(text.count("data:image/png;base64,"), 4)
            self.assertIn("How to read this", text)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("frequency_hz", header)
            self.assertGreater(html_path.stat().st_size, 10_000)

    def test_noise_only_axis_is_not_a_natural_frequency(self):
        rng = np.random.default_rng(1)
        accel = {axis: values.copy() for axis, values in self.recording.accel.items()}
        accel["y"] = rng.normal(0.0, 0.02, size=self.recording.force.shape)
        noisy = recording_from_channels(
            self.recording.time_s,
            self.recording.force.copy(),
            accel,
            sample_rate_hz=self.recording.sample_rate_hz,
        )
        result = analyze_recording(noisy, AnalysisSettings())
        freqs = [mode.frequency_hz for mode in result.modes]
        self.assertTrue(any(98.0 <= freq <= 108.0 for freq in freqs), freqs)
        self.assertTrue(all("y" not in mode.axes for mode in result.modes), result.modes)


class HeaderTests(unittest.TestCase):
    def test_parentheses_units_and_comment_preamble(self):
        text = (
            "# a comment\n"
            "\n"
            "Operator, Chris\n"
            "Time (s), Force (N), Accel X (g), Accel Y (g), Accel Z (g)\n"
            "0, 0, 0, 0, 0\n"
            "0.1, 1, 0.2, 0.1, 0.05\n"
        )
        table, mapping, recording = _load_text(text)
        self.assertEqual(table.headers[0], "Time (s)")
        self.assertTrue(mapping.confident)
        self.assertEqual(recording.force_unit, "N")
        self.assertEqual(recording.accel_unit["z"], "g")
        self.assertEqual(mapping.x, 2)

    def test_units_row_under_the_names(self):
        text = "Time, Force, Accel X, Accel Y, Accel Z\ns, N, g, g, g\n0, 0, 0, 0, 0\n0.1, 1, 0, 0, 0\n"
        _table, mapping, recording = _load_text(text)
        self.assertTrue(mapping.confident)
        self.assertEqual(recording.force_unit, "N")
        self.assertEqual(recording.accel_unit["y"], "g")

    def test_generic_channel_names_stay_unassigned(self):
        text = (
            "Channel 1, Channel 2, Channel 3, Channel 4, Channel 5\n"
            "0, 0, 0, 0, 0\n"
            "0.1, 1, 0, 0, 0\n"
        )
        _table, mapping, _recording = _load_text(text, build=False)
        self.assertFalse(mapping.confident)
        self.assertFalse(mapping.assumed_order)
        self.assertIsNone(mapping.force)
        self.assertIsNone(mapping.x)

    def test_headerless_five_columns_assume_order(self):
        text = "0, 0, 0, 0, 0\n0.1, 1, 0.2, 0.1, 0.05\n"
        _table, mapping, recording = _load_text(text)
        self.assertTrue(mapping.assumed_order)
        self.assertFalse(mapping.confident)
        self.assertEqual((mapping.time, mapping.force, mapping.x, mapping.y, mapping.z), (0, 1, 2, 3, 4))
        self.assertTrue(any("Assuming order" in note for note in recording.notes))


class FaultTests(unittest.TestCase):
    def test_double_hit_is_rejected(self):
        fs = 5120.0
        n = int(3 * fs)
        time_s = np.arange(n) / fs
        rng = np.random.default_rng(2)
        force = rng.normal(0.0, 0.05, n)
        _pulse(force, fs, 0.40, 0.001, 120.0)
        _pulse(force, fs, 1.60, 0.001, 100.0)
        _pulse(force, fs, 1.608, 0.001, 70.0)
        recording = recording_from_channels(
            time_s,
            force,
            {"x": rng.normal(0.0, 0.01, n)},
            sample_rate_hz=fs,
        )
        result = analyze_recording(recording, AnalysisSettings())
        accepted = result.accepted_hits
        rejected = [hit for hit in result.hits if not hit.accepted]
        self.assertEqual(len(accepted), 1, result.hits)
        self.assertEqual(len(rejected), 1, result.hits)
        self.assertEqual(rejected[0].reason, "double-hit")
        self.assertLess(abs(accepted[0].peak_time_s - 0.40), 0.002)

    def test_flat_topped_channel_is_flagged(self):
        fs = 5120.0
        n = int(2 * fs)
        time_s = np.arange(n) / fs
        rng = np.random.default_rng(3)
        force = rng.normal(0.0, 0.05, n)
        _pulse(force, fs, 0.5, 0.001, 80.0)
        accel = rng.normal(0.0, 0.01, n)
        accel[20:28] = 80.0
        recording = recording_from_channels(time_s, force, {"x": accel}, sample_rate_hz=fs)
        result = analyze_recording(recording, AnalysisSettings())
        self.assertTrue(any("overloaded" in warning.lower() for warning in result.warnings), result.warnings)


def _load_text(text: str, build: bool = True):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.csv"
        path.write_text(text, encoding="utf-8")
        table = read_table(path)
        mapping = propose_mapping(table)
        recording = build_recording(table, mapping) if build else None
        return table, mapping, recording


def _pulse(force: np.ndarray, fs: float, t_peak: float, width: float, amplitude: float) -> None:
    idx = np.arange(force.size)
    phase = (idx / fs - (t_peak - width / 2.0)) / width
    keep = (phase >= 0.0) & (phase <= 1.0)
    force[keep] += amplitude * np.sin(np.pi * phase[keep])


class WindowTests(unittest.TestCase):
    def test_window_opens_on_the_example(self):
        if not os.environ.get("DISPLAY"):
            self.skipTest("No display, so the desktop window was not opened.")
        if not EXAMPLE.exists():
            self.skipTest("Example recording is missing.")
        script = """
import app
application = app.launch(load_example=True, settings_path="/tmp/bracket-modes-test-settings.json", interactive=False)
application.root.update_idletasks()
application.root.update()
modes = application.analysis.modes
print("GUI_OK", len(application.analysis.accepted_hits), modes[0].frequency_hz)
application.root.destroy()
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
        self.assertIn("GUI_OK 10", completed.stdout)


if __name__ == "__main__":
    unittest.main()
