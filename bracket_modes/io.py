"""Load an impulse-test CSV and map its columns."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_UNIT_TOKEN = re.compile(
    r"^(s|sec|secs|seconds|n|kn|lbf|g|v|mv|volts?|m/s\^?2|ms2)$",
    re.IGNORECASE,
)
_UNIT_PRETTY = {
    "n": "N",
    "kn": "kN",
    "lbf": "lbf",
    "g": "g",
    "v": "V",
    "mv": "mV",
    "volt": "V",
    "volts": "V",
    "ms2": "m/s^2",
    "m/s2": "m/s^2",
    "m/s^2": "m/s^2",
}


@dataclass
class ColumnMapping:
    time: int | None = None
    force: int | None = None
    x: int | None = None
    y: int | None = None
    z: int | None = None
    confident: bool = False
    assumed_order: bool = False
    note: str = ""

    def index_for(self, axis: str) -> int | None:
        return {"x": self.x, "y": self.y, "z": self.z}[axis]

    def assigned_indices(self) -> list[int]:
        values = [self.time, self.force, self.x, self.y, self.z]
        return [v for v in values if v is not None]


@dataclass
class Table:
    headers: list[str] | None
    data: np.ndarray
    source_name: str
    source_path: str

    @property
    def n_columns(self) -> int:
        return int(self.data.shape[1]) if self.data.ndim == 2 else 0

    def column_labels(self) -> list[str]:
        if self.headers:
            return [f"{i + 1}: {name}" for i, name in enumerate(self.headers)]
        return [f"Column {i + 1}" for i in range(self.n_columns)]

    def signature(self) -> str:
        if not self.headers:
            return f"headerless:{self.n_columns}"
        return "|".join(self.headers)


@dataclass
class Recording:
    time_s: np.ndarray
    force: np.ndarray
    accel: dict[str, np.ndarray]
    sample_rate_hz: float
    force_unit: str
    accel_unit: dict[str, str]
    source_name: str
    headers: list[str] | None
    mapping: ColumnMapping
    resampled: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if self.time_s.size == 0:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])


def read_table(path: str | Path) -> Table:
    """Read a CSV, skipping comments and a short metadata preamble."""
    path = Path(path)
    try:
        raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not open {path.name}: {exc}") from exc

    logical: list[tuple[int, list[str]]] = []
    for lineno, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cells = [cell.strip() for cell in next(csv.reader([stripped]))]
        if not any(cells):
            continue
        logical.append((lineno, cells))
    if not logical:
        raise ValueError("The file is empty.")

    numeric_at = next((i for i, (_, cells) in enumerate(logical) if _is_numeric(cells)), None)
    if numeric_at is None:
        raise ValueError("No rows of measurements were found in the file.")

    headers: list[str] | None = None
    if numeric_at >= 1:
        headers = list(logical[numeric_at - 1][1])
        if numeric_at >= 2:
            names = logical[numeric_at - 2][1]
            units = headers
            if (
                len(names) == len(units)
                and _looks_like_units(units)
                and not _is_numeric(names)
            ):
                headers = [
                    f"{name} ({unit})" if unit and "(" not in name else name
                    for name, unit in zip(names, units)
                ]

    first_data_line = logical[numeric_at][0]
    try:
        data = np.loadtxt(
            path,
            delimiter=",",
            skiprows=first_data_line,
            comments="#",
            ndmin=2,
            encoding="utf-8-sig",
        )
    except (ValueError, OSError) as exc:
        raise ValueError(
            "Could not read the measurements. Look for blank lines or text in the data rows."
        ) from exc
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("The file needs at least two rows and two columns of numbers.")
    if headers is not None and len(headers) != data.shape[1]:
        raise ValueError("The column names and the data have different numbers of columns.")
    if not np.isfinite(data).all():
        raise ValueError("The file has a measurement that is not a finite number.")
    return Table(
        headers=headers,
        data=data,
        source_name=path.name,
        source_path=str(path),
    )


def propose_mapping(table: Table) -> ColumnMapping:
    """Guess time, force, and X/Y/Z. Generic names are left blank on purpose."""
    n_cols = table.n_columns
    if table.headers is None:
        if n_cols == 5:
            return ColumnMapping(
                time=0,
                force=1,
                x=2,
                y=3,
                z=4,
                confident=False,
                assumed_order=True,
                note="No column names were found. Assuming order: time, force, X, Y, Z.",
            )
        return ColumnMapping(
            confident=False,
            note="No column names were found. Choose time, force, and the acceleration columns.",
        )

    parsed = [_split_header(name) for name in table.headers]
    time_scores = [_score_time(parts) for parts, _unit in parsed]
    force_scores = [_score_force(parts) for parts, _unit in parsed]
    axis_scores = {
        axis: [_score_axis(parts, axis) for parts, _unit in parsed] for axis in ("x", "y", "z")
    }
    time_idx, time_ok = _pick(time_scores)
    force_idx, force_ok = _pick(force_scores)
    axis_idx = {}
    axis_ok = {}
    for axis in ("x", "y", "z"):
        axis_idx[axis], axis_ok[axis] = _pick(axis_scores[axis])

    notes: list[str] = []
    if max(time_scores) > 0 and not time_ok:
        notes.append("More than one column looks like time. Choose the time column.")
        time_idx = None
    if max(force_scores) > 0 and not force_ok:
        notes.append("More than one column looks like force. Choose the force column.")
        force_idx = None
    for axis in ("x", "y", "z"):
        if max(axis_scores[axis]) > 0 and not axis_ok[axis]:
            notes.append(f"More than one column looks like {axis.upper()}. Choose it.")
            axis_idx[axis] = None

    chosen = [time_idx, force_idx, axis_idx["x"], axis_idx["y"], axis_idx["z"]]
    if _has_duplicates([c for c in chosen if c is not None]):
        notes.append("Two measurements were matched to the same column. Choose the columns.")
        time_idx = force_idx = None
        axis_idx = {"x": None, "y": None, "z": None}
        time_ok = force_ok = False
        axis_ok = {"x": False, "y": False, "z": False}

    any_axis = any(axis_idx[axis] is not None for axis in ("x", "y", "z"))
    confident = bool(time_ok and force_ok and any_axis and not notes)
    if not any_axis and not notes:
        notes.append("Choose which columns are time, force, and acceleration.")
    return ColumnMapping(
        time=time_idx,
        force=force_idx,
        x=axis_idx["x"],
        y=axis_idx["y"],
        z=axis_idx["z"],
        confident=confident,
        assumed_order=False,
        note=" ".join(notes),
    )


def build_recording(
    table: Table,
    mapping: ColumnMapping,
    sample_rate_hz: float | None = None,
) -> Recording:
    """Slice the chosen columns into a recording. Raises ValueError with a plain reason."""
    n_cols = table.n_columns
    chosen = mapping.assigned_indices()
    if any(idx < 0 or idx >= n_cols for idx in chosen):
        raise ValueError("A chosen column is not in this file.")
    if _has_duplicates(chosen):
        raise ValueError("Two fields are set to the same column.")
    if mapping.force is None:
        raise ValueError("Choose the force column.")
    axes = [axis for axis in ("x", "y", "z") if mapping.index_for(axis) is not None]
    if not axes:
        raise ValueError("Choose at least one acceleration column.")

    data = table.data
    notes = []
    if mapping.note and (mapping.assumed_order or not mapping.confident):
        notes.append(mapping.note)

    parsed_units = []
    if table.headers:
        parsed_units = [_split_header(name)[1] for name in table.headers]
    else:
        parsed_units = [""] * n_cols

    resampled = False
    if mapping.time is not None:
        time_s = np.asarray(data[:, mapping.time], dtype=float)
        if np.any(np.diff(time_s) <= 0):
            raise ValueError("The time column must increase on every row.")
        dt = np.diff(time_s)
        med = float(np.median(dt))
        if med <= 0:
            raise ValueError("The time column must increase on every row.")
        sample_rate = 1.0 / med
        if float(np.max(np.abs(dt - med))) > 0.001 * med:
            t_new = np.arange(time_s[0], time_s[-1], med)
            if t_new.size < 2:
                raise ValueError("The time column is too short to resample.")
            data = np.column_stack(
                [np.interp(t_new, time_s, data[:, col]) for col in range(n_cols)]
            )
            time_s = t_new
            resampled = True
            sample_rate = 1.0 / med
            notes.append(
                "The time steps were uneven by more than 0.1%, so the recording was resampled onto an even grid."
            )
        if sample_rate < 50:
            notes.append(
                "The sample rate came out below 50 Hz. If the time column is really sample numbers, clear it and type the sample rate in hertz."
            )
    else:
        if sample_rate_hz is None or sample_rate_hz <= 0:
            raise ValueError("Choose the time column, or type the sample rate in hertz.")
        sample_rate = float(sample_rate_hz)
        time_s = np.arange(data.shape[0], dtype=float) / sample_rate

    force = np.asarray(data[:, mapping.force], dtype=float).copy()
    accel = {
        axis: np.asarray(data[:, mapping.index_for(axis)], dtype=float).copy() for axis in axes
    }
    force_unit = parsed_units[mapping.force] if mapping.force is not None else ""
    accel_unit = {axis: parsed_units[mapping.index_for(axis)] for axis in axes}
    return Recording(
        time_s=time_s,
        force=force,
        accel=accel,
        sample_rate_hz=float(sample_rate),
        force_unit=force_unit,
        accel_unit=accel_unit,
        source_name=table.source_name,
        headers=list(table.headers) if table.headers else None,
        mapping=mapping,
        resampled=resampled,
        notes=notes,
    )


def recording_from_channels(
    time_s: np.ndarray,
    force: np.ndarray,
    accel: dict[str, np.ndarray],
    *,
    sample_rate_hz: float | None = None,
    name: str = "memory",
    force_unit: str = "N",
    accel_unit: str = "g",
) -> Recording:
    """Build a recording from arrays already in memory. Used by tests and the example generator."""
    time_s = np.asarray(time_s, dtype=float)
    force = np.asarray(force, dtype=float)
    accel = {key: np.asarray(value, dtype=float) for key, value in accel.items()}
    if sample_rate_hz is None:
        sample_rate_hz = float(1.0 / np.median(np.diff(time_s)))
    return Recording(
        time_s=time_s,
        force=force,
        accel=accel,
        sample_rate_hz=float(sample_rate_hz),
        force_unit=force_unit,
        accel_unit={key: accel_unit for key in accel},
        source_name=name,
        headers=["time_s", "force", *[f"accel_{key}" for key in accel]],
        mapping=ColumnMapping(confident=True),
    )


def _is_numeric(cells: list[str]) -> bool:
    if len(cells) < 2:
        return False
    for cell in cells:
        try:
            float(cell)
        except ValueError:
            return False
    return True


def _looks_like_units(cells: list[str]) -> bool:
    return bool(cells) and all(_UNIT_TOKEN.match(cell.strip() or "") for cell in cells)


def _split_header(name: str) -> tuple[list[str], str]:
    raw = name.strip().lstrip("\ufeff")
    unit = ""
    match = re.search(r"\(([^)]+)\)\s*$", raw)
    if match:
        unit = _pretty_unit(match.group(1))
        raw = raw[: match.start()]
    raw = raw.strip().lower().replace("-", " ").replace("/", " ")
    parts = [part for part in re.split(r"[\s_]+", raw) if part]
    if not unit and parts:
        pretty = _pretty_unit(parts[-1])
        if pretty and parts[-1].lower() in _UNIT_PRETTY:
            parts = parts[:-1]
            unit = pretty
    return parts, unit


def _pretty_unit(unit: str) -> str:
    text = unit.strip()
    if not text:
        return ""
    mapped = _UNIT_PRETTY.get(text.lower())
    if mapped:
        return mapped
    if _UNIT_TOKEN.match(text) and text.lower() in {"s", "sec", "secs", "seconds"}:
        return "s"
    return text


def _score_time(parts: list[str]) -> int:
    if not parts:
        return 0
    if "time" in parts or "timestamp" in parts:
        return 3
    if parts in (["t"], ["sec"], ["secs"], ["seconds"]):
        return 3
    return 0


def _score_force(parts: list[str]) -> int:
    if not parts:
        return 0
    if any(part in {"force", "hammer", "impact", "load", "excitation", "input"} for part in parts):
        return 3
    if parts == ["f"]:
        return 2
    return 0


def _score_axis(parts: list[str], axis: str) -> int:
    tokens = {axis, f"a{axis}", f"acc{axis}", f"accel{axis}"}
    if not any(part in tokens for part in parts):
        return 0
    if any(part in {"accel", "acc", "acceleration", "triax", "tri"} for part in parts):
        return 3
    if any(part.startswith("acc") for part in parts):
        return 3
    return 2


def _pick(scores: list[int]) -> tuple[int | None, bool]:
    if not scores or max(scores) <= 0:
        return None, False
    best = max(scores)
    chosen = [i for i, score in enumerate(scores) if score == best]
    if len(chosen) != 1:
        return None, False
    return chosen[0], True


def _has_duplicates(values: list[int]) -> bool:
    return len(values) != len(set(values))
