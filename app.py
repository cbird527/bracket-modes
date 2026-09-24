#!/usr/bin/env python3
"""Desktop tool for natural frequencies from an impulse-hammer CSV.

Run from this folder:

    python3 app.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from bracket_modes.analyze import AnalysisSettings, analyze_recording, settings_error, summary_sentence
from bracket_modes.io import ColumnMapping, Table, build_recording, propose_mapping, read_table
from bracket_modes.plots import draw_coherence, draw_frf, draw_hit, draw_recording
from bracket_modes.report import write_report

APP_DIR = Path(__file__).resolve().parent
EXAMPLE_CSV = APP_DIR / "examples" / "bracket_10hits.csv"
DEFAULT_SETTINGS = APP_DIR / "bracket_modes_settings.json"
NONE = "(none)"


class App:
    def __init__(
        self,
        load_example: bool = False,
        settings_path: Path | None = None,
        interactive: bool = True,
    ):
        self.settings_path = Path(settings_path) if settings_path else DEFAULT_SETTINGS
        self.interactive = interactive
        self.table: Table | None = None
        self.recording = None
        self.analysis = None
        self._mappings: dict = {}
        self._ready = False
        self._after_id = None
        self._label_to_index: dict[str, int] = {}

        self.root = tk.Tk()
        self.root.title("Bracket natural frequencies")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)

        self.threshold = tk.IntVar(value=30)
        self.pre_var = tk.StringVar(value="20")
        self.block_var = tk.StringVar(value="1.0")
        self.fmin_var = tk.StringVar(value="10")
        self.fmax_var = tk.StringVar(value="")
        self.coh_var = tk.StringVar(value="0.85")
        self.fs_var = tk.StringVar(value="")
        self.expo_var = tk.BooleanVar(value=True)
        self.combined_var = tk.BooleanVar(value=True)
        self.time_var = tk.StringVar(value=NONE)
        self.force_var = tk.StringVar(value=NONE)
        self.axis_vars = {axis: tk.StringVar(value=NONE) for axis in ("x", "y", "z")}
        self.summary_var = tk.StringVar(value="Open a CSV of the hammer test, or open the example.")
        self.status_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        self.fs_hint = tk.StringVar(value="Used only when no time column is chosen.")

        self._build()
        self._load_settings_file()
        self._ready = True
        if load_example:
            self.open_path(EXAMPLE_CSV)

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=(8, 8, 8, 0))
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open CSV", command=self._open_dialog).pack(side="left")
        ttk.Button(top, text="Open example", command=self._open_example).pack(side="left", padx=(6, 0))
        self.export_btn = ttk.Button(top, text="Export report", command=self._export)
        self.export_btn.pack(side="left", padx=(6, 0))
        self.export_btn.state(["disabled"])
        ttk.Label(top, textvariable=self.summary_var, font=("TkDefaultFont", 12, "bold")).pack(
            side="left", padx=(16, 0)
        )

        left = ttk.Frame(self.root, padding=(8, 8))
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Channels", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self.combos = {}
        labels = (("Time", self.time_var), ("Force", self.force_var))
        for row, (text, var) in enumerate(labels, start=1):
            ttk.Label(left, text=text).grid(row=row, column=0, sticky="w", pady=2)
            combo = ttk.Combobox(left, textvariable=var, state="readonly", width=22, values=(NONE,))
            combo.grid(row=row, column=1, sticky="ew", pady=2)
            combo.bind("<<ComboboxSelected>>", self._schedule)
            self.combos[text.lower()] = combo
        for row, axis in enumerate(("x", "y", "z"), start=3):
            ttk.Label(left, text=axis.upper()).grid(row=row, column=0, sticky="w", pady=2)
            combo = ttk.Combobox(
                left, textvariable=self.axis_vars[axis], state="readonly", width=22, values=(NONE,)
            )
            combo.grid(row=row, column=1, sticky="ew", pady=2)
            combo.bind("<<ComboboxSelected>>", self._schedule)
            self.combos[axis] = combo
        ttk.Label(left, text="Sample rate (Hz)").grid(row=6, column=0, sticky="w", pady=2)
        fs_entry = ttk.Entry(left, textvariable=self.fs_var, width=12)
        fs_entry.grid(row=6, column=1, sticky="ew", pady=2)
        fs_entry.bind("<KeyRelease>", self._schedule)
        fs_entry.bind("<Return>", self._schedule)
        self.fs_entry = fs_entry
        ttk.Label(left, textvariable=self.fs_hint, wraplength=250).grid(
            row=7, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(left, textvariable=self.note_var, wraplength=250, foreground="#8a5a00").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(4, 8)
        )

        ttk.Separator(left).grid(row=9, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(left, text="Settings", font=("TkDefaultFont", 11, "bold")).grid(
            row=10, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(left, text="Hit threshold (%)").grid(row=11, column=0, columnspan=2, sticky="w")
        scale = tk.Scale(
            left,
            from_=5,
            to=80,
            orient="horizontal",
            variable=self.threshold,
            command=self._schedule,
            showvalue=True,
        )
        scale.grid(row=12, column=0, columnspan=2, sticky="ew")
        self._entry_row(left, 13, "Time before hit (ms)", self.pre_var)
        self._entry_row(left, 14, "Time after hit (s)", self.block_var)
        self._entry_row(left, 15, "Lower frequency (Hz)", self.fmin_var)
        self._entry_row(left, 16, "Upper frequency (Hz)", self.fmax_var)
        ttk.Label(left, text="Blank upper frequency searches to 40% of the sample rate.", wraplength=250).grid(
            row=17, column=0, columnspan=2, sticky="w"
        )
        self._entry_row(left, 18, "Coherence minimum", self.coh_var)
        ttk.Checkbutton(
            left,
            text="Exponential window",
            variable=self.expo_var,
            command=self._schedule,
        ).grid(row=19, column=0, columnspan=2, sticky="w", pady=(6, 0))
        left.columnconfigure(1, weight=1)

        right = ttk.Frame(self.root, padding=(0, 8, 8, 0))
        right.pack(side="left", fill="both", expand=True)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        self.fig_rec = Figure(figsize=(7.2, 5.4), dpi=100)
        self.fig_hit = Figure(figsize=(7.2, 5.0), dpi=100)
        self.fig_frf = Figure(figsize=(7.2, 4.3), dpi=100)
        self.fig_coh = Figure(figsize=(7.2, 3.3), dpi=100)
        self.canvas_rec = self._add_figure_tab("Recording", self.fig_rec)
        self.canvas_hit = self._add_figure_tab("One blow", self.fig_hit)
        freq = ttk.Frame(self.notebook)
        self.notebook.add(freq, text="Frequencies")
        ttk.Checkbutton(
            freq,
            text="Show combined response",
            variable=self.combined_var,
            command=self._redraw_frf,
        ).pack(side="top", anchor="w")
        columns = ("frequency", "axes", "strongest", "coherence", "damping", "quality")
        self.tree = ttk.Treeview(freq, columns=columns, show="headings", height=6)
        headings = (
            ("frequency", "Frequency", 110),
            ("axes", "Axes", 80),
            ("strongest", "Strongest axis", 110),
            ("coherence", "Coherence", 90),
            ("damping", "Approximate damping", 150),
            ("quality", "Quality", 70),
        )
        for key, text, width in headings:
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor="w")
        self.tree.pack(side="bottom", fill="x", padx=4, pady=(0, 6))
        self.canvas_frf = self._embed_figure(freq, self.fig_frf)

        quality = ttk.Frame(self.notebook)
        self.notebook.add(quality, text="Quality")
        self.quality_text = tk.Text(quality, height=8, wrap="word")
        self.quality_text.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        self.quality_text.configure(state="disabled")
        self.canvas_coh = self._embed_figure(quality, self.fig_coh)

        ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 4)).pack(
            side="bottom", fill="x"
        )
        draw_recording_placeholder(self.fig_rec)
        self.canvas_rec.draw_idle()

    def _add_figure_tab(self, title: str, fig: Figure) -> FigureCanvasTkAgg:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        return self._embed_figure(frame, fig)

    def _embed_figure(self, frame: ttk.Frame, fig: Figure) -> FigureCanvasTkAgg:
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        return canvas

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=12)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        entry.bind("<KeyRelease>", self._schedule)
        entry.bind("<Return>", self._schedule)
        entry.bind("<FocusOut>", self._schedule)

    def _open_dialog(self) -> None:
        start = str(EXAMPLE_CSV.parent if EXAMPLE_CSV.parent.exists() else APP_DIR)
        path = filedialog.askopenfilename(
            initialdir=start,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.open_path(Path(path))

    def _open_example(self) -> None:
        if not EXAMPLE_CSV.exists():
            self._error(
                "Example missing",
                "The example recording is not in the examples folder.\n"
                "From this folder, run: python3 -m bracket_modes.synth",
            )
            return
        self.open_path(EXAMPLE_CSV)

    def open_path(self, path: Path) -> None:
        try:
            self.table = read_table(path)
        except ValueError as exc:
            self._error("Could not read the CSV", str(exc))
            return
        saved = self._saved_mapping(self.table.signature())
        mapping = saved if saved is not None else propose_mapping(self.table)
        self._apply_mapping(mapping)
        self._analyze()

    def _export(self) -> None:
        if self.recording is None or self.analysis is None:
            self._error("Nothing to export", "Open a recording first.")
            return
        stem = Path(self.recording.source_name).stem + "_report.html"
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            initialfile=stem,
            filetypes=[("HTML report", "*.html")],
        )
        if not path:
            return
        csv_path = write_report(path, self.recording, self.analysis)
        self.status_var.set(f"Saved {path} and {csv_path.name}")

    def _schedule(self, *_args) -> None:
        if not self._ready or self.table is None:
            return
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(250, self._analyze)

    def _analyze(self) -> None:
        self._after_id = None
        if self.table is None:
            return
        try:
            settings = self._settings_from_form()
            problem = settings_error(settings)
            if problem:
                self.status_var.set(problem)
                return
            mapping = self._mapping_from_form()
            sample_rate = None
            if mapping.time is None:
                text = self.fs_var.get().strip().replace(",", "")
                if not text:
                    self.status_var.set("Choose the time column, or type the sample rate in hertz.")
                    self.note_var.set(mapping.note)
                    return
                sample_rate = float(text)
            recording = build_recording(self.table, mapping, sample_rate)
            analysis = analyze_recording(recording, settings)
        except ValueError as exc:
            text = str(exc)
            if "could not convert" in text or "invalid literal" in text:
                text = "Check the settings. Frequencies, times, and coherence need to be plain numbers."
            self.status_var.set(text)
            return
        except Exception as exc:  # unexpected; keep the traceback in the terminal
            traceback.print_exc()
            self.status_var.set(str(exc))
            if not self.interactive:
                raise
            return
        self.recording = recording
        self.analysis = analysis
        self._refresh()
        self._save_settings()

    def _refresh(self) -> None:
        assert self.recording is not None and self.analysis is not None
        analysis = self.analysis
        self.summary_var.set(summary_sentence(analysis.modes))
        n_used = len(analysis.accepted_hits)
        n_left = len(analysis.hits) - n_used
        resolution = f"{analysis.df_hz:.2f} Hz resolution" if analysis.df_hz else "no spectrum"
        self.status_var.set(
            f"{analysis.sample_rate_hz:.1f} Hz · {analysis.duration_s:.2f} s · "
            f"{n_used} blows used, {n_left} left out · {resolution}"
        )
        if self._mapping_from_form().time is not None:
            self.fs_entry.state(["disabled"])
            self.fs_hint.set(f"Using {analysis.sample_rate_hz:.2f} Hz from the time column.")
        else:
            self.fs_entry.state(["!disabled"])
            self.fs_hint.set("Using the sample rate typed above.")
        self.note_var.set(self.recording.mapping.note)
        draw_recording(self.fig_rec, self.recording, analysis)
        draw_hit(self.fig_hit, self.recording, analysis)
        draw_frf(self.fig_frf, self.recording, analysis, show_combined=self.combined_var.get())
        draw_coherence(self.fig_coh, analysis)
        for canvas in (self.canvas_rec, self.canvas_hit, self.canvas_frf, self.canvas_coh):
            canvas.draw_idle()
        self._fill_table()
        self._fill_quality()
        self.export_btn.state(["!disabled"])

    def _redraw_frf(self) -> None:
        if self.recording is None or self.analysis is None:
            return
        draw_frf(self.fig_frf, self.recording, self.analysis, show_combined=self.combined_var.get())
        self.canvas_frf.draw_idle()

    def _fill_table(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        for mode in self.analysis.modes:
            coherence = "—" if mode.coherence is None else f"{mode.coherence:.2f}"
            if mode.damping_reliable and mode.damping_ratio is not None:
                damping = f"{mode.damping_ratio * 100:.2f}%"
            else:
                damping = "not reliable"
            self.tree.insert(
                "",
                "end",
                values=(
                    f"{mode.frequency_hz:.1f} Hz",
                    ", ".join(axis.upper() for axis in mode.axes),
                    mode.strongest_axis.upper(),
                    coherence,
                    damping,
                    "Good" if mode.quality == "good" else "Check",
                ),
            )

    def _fill_quality(self) -> None:
        lines = []
        if self.analysis.warnings:
            lines.append("Check these")
            lines.extend(f"• {item}" for item in self.analysis.warnings)
            lines.append("")
        else:
            lines.append("No measurement problems were flagged.")
            lines.append("")
        lines.append("What was done")
        lines.extend(f"• {item}" for item in self.analysis.notes)
        self.quality_text.configure(state="normal")
        self.quality_text.delete("1.0", "end")
        self.quality_text.insert("1.0", "\n".join(lines))
        self.quality_text.configure(state="disabled")

    def _settings_from_form(self) -> AnalysisSettings:
        fmax_text = self.fmax_var.get().strip().replace(",", "")
        fmax = float(fmax_text) if fmax_text else None
        return AnalysisSettings(
            threshold_fraction=float(self.threshold.get()) / 100.0,
            pretrigger_s=float(self.pre_var.get()) / 1000.0,
            block_length_s=float(self.block_var.get()),
            fmin_hz=float(self.fmin_var.get()),
            fmax_hz=fmax,
            coherence_min=float(self.coh_var.get()),
            exponential_window=bool(self.expo_var.get()),
        )

    def _mapping_from_form(self) -> ColumnMapping:
        proposed = propose_mapping(self.table) if self.table is not None else ColumnMapping()
        chosen = ColumnMapping(
            time=self._label_to_index.get(self.time_var.get()),
            force=self._label_to_index.get(self.force_var.get()),
            x=self._label_to_index.get(self.axis_vars["x"].get()),
            y=self._label_to_index.get(self.axis_vars["y"].get()),
            z=self._label_to_index.get(self.axis_vars["z"].get()),
        )
        same = (
            chosen.time == proposed.time
            and chosen.force == proposed.force
            and chosen.x == proposed.x
            and chosen.y == proposed.y
            and chosen.z == proposed.z
        )
        if same:
            chosen.confident = proposed.confident
            chosen.assumed_order = proposed.assumed_order
            chosen.note = proposed.note
        else:
            chosen.confident = True
            chosen.note = ""
        return chosen

    def _apply_mapping(self, mapping: ColumnMapping) -> None:
        assert self.table is not None
        labels = [NONE] + self.table.column_labels()
        # column_labels are "1: name"; the index in that list is the column index.
        self._label_to_index = {label: index for index, label in enumerate(self.table.column_labels())}
        for combo in self.combos.values():
            combo.configure(values=labels)
        self.time_var.set(_label_for(self.table, mapping.time))
        self.force_var.set(_label_for(self.table, mapping.force))
        for axis in ("x", "y", "z"):
            self.axis_vars[axis].set(_label_for(self.table, mapping.index_for(axis)))
        self.note_var.set(mapping.note)

    def _saved_mapping(self, signature: str) -> ColumnMapping | None:
        raw = self._mappings.get(signature)
        if not isinstance(raw, dict) or self.table is None:
            return None
        mapping = ColumnMapping(
            time=_optional_index(raw.get("time")),
            force=_optional_index(raw.get("force")),
            x=_optional_index(raw.get("x")),
            y=_optional_index(raw.get("y")),
            z=_optional_index(raw.get("z")),
            confident=True,
        )
        for index in mapping.assigned_indices():
            if index < 0 or index >= self.table.n_columns:
                return None
        if mapping.force is None or not any(mapping.index_for(axis) is not None for axis in ("x", "y", "z")):
            return None
        if len(mapping.assigned_indices()) != len(set(mapping.assigned_indices())):
            return None
        return mapping

    def _load_settings_file(self) -> None:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        self._mappings = data.get("mappings", {}) if isinstance(data.get("mappings"), dict) else {}
        self._set_var(self.threshold, data.get("threshold_percent"), integer=True)
        self._set_text(self.pre_var, data.get("pretrigger_ms"))
        self._set_text(self.block_var, data.get("block_length_s"))
        self._set_text(self.fmin_var, data.get("fmin_hz"))
        if "fmax_hz" in data:
            self.fmax_var.set("" if data["fmax_hz"] in (None, "") else str(data["fmax_hz"]))
        self._set_text(self.coh_var, data.get("coherence_min"))
        if isinstance(data.get("exponential_window"), bool):
            self.expo_var.set(data["exponential_window"])

    def _save_settings(self) -> None:
        if self.table is None:
            return
        mapping = self._mapping_from_form()
        payload = {
            "threshold_percent": int(self.threshold.get()),
            "pretrigger_ms": self.pre_var.get().strip(),
            "block_length_s": self.block_var.get().strip(),
            "fmin_hz": self.fmin_var.get().strip(),
            "fmax_hz": self.fmax_var.get().strip() or None,
            "coherence_min": self.coh_var.get().strip(),
            "exponential_window": bool(self.expo_var.get()),
            "mappings": self._mappings,
        }
        payload["mappings"][self.table.signature()] = {
            "time": mapping.time,
            "force": mapping.force,
            "x": mapping.x,
            "y": mapping.y,
            "z": mapping.z,
        }
        try:
            self.settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _set_text(self, var: tk.StringVar, value) -> None:
        if value is not None and value != "":
            var.set(str(value))

    def _set_var(self, var: tk.IntVar, value, integer: bool = False) -> None:
        if isinstance(value, (int, float)) and integer:
            var.set(int(value))

    def _error(self, title: str, text: str) -> None:
        if self.interactive:
            messagebox.showerror(title, text)
        else:
            raise RuntimeError(text)


def _label_for(table: Table, index: int | None) -> str:
    if index is None:
        return NONE
    labels = table.column_labels()
    if index < 0 or index >= len(labels):
        return NONE
    return labels[index]


def _optional_index(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def draw_recording_placeholder(fig: Figure) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        "Open a CSV to see the hammer blows.",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()


def launch(
    load_example: bool = False,
    settings_path: Path | None = None,
    interactive: bool = True,
) -> App:
    return App(load_example=load_example, settings_path=settings_path, interactive=interactive)


def main() -> None:
    app = launch()
    app.root.mainloop()


if __name__ == "__main__":
    main()
