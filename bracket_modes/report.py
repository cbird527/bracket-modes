"""Self-contained HTML report and a CSV of the frequencies."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from html import escape
from pathlib import Path

from matplotlib.figure import Figure

from bracket_modes.analyze import Analysis, summary_sentence
from bracket_modes.io import Recording
from bracket_modes.plots import draw_coherence, draw_frf, draw_hit, draw_recording

_HOW_TO_READ = (
    "The frequencies above are the rates the bracket wants to ring at. "
    "On the response chart, the bold lines are the average of the blows and the faint lines are the individual blows. "
    "A peak is a natural frequency. Coherence near 1 means the blows agreed with each other; "
    "peaks that were not repeatable are left off the list. "
    "The damping percentage is an approximation from how wide the peak is. "
    "“Not reliable” means a neighbor, or the analysis window, got in the way of that width. "
    "This is one accelerometer location. It does not show the shape of the motion."
)


def write_report(path: str | Path, recording: Recording, analysis: Analysis) -> Path:
    """Write the HTML report and a sibling ``*_frequencies.csv``. Returns the CSV path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [
        ("Hammer hits found in the recording", _png(draw_recording, (8.8, 7.2), recording, analysis)),
        ("One blow, and the parts of the signal that are used", _png(draw_hit, (8.8, 6.4), recording, analysis)),
        ("How strongly the bracket responded", _png(draw_frf, (8.8, 5.2), recording, analysis)),
        ("How repeatable each frequency was", _png(draw_coherence, (8.8, 4.2), analysis)),
    ]
    csv_path = path.with_name(f"{path.stem}_frequencies.csv")
    _write_frequency_csv(csv_path, analysis)
    path.write_text(_html(recording, analysis, images, csv_path.name), encoding="utf-8")
    return csv_path


def _png(draw, size: tuple[float, float], *args) -> str:
    fig = Figure(figsize=size, dpi=120)
    draw(fig, *args)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor="white")
    fig.clear()
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_frequency_csv(path: Path, analysis: Analysis) -> None:
    lines = [
        "frequency_hz,axes,strongest_axis,coherence,damping_ratio,damping_reliable,quality,prominence_db"
    ]
    for mode in analysis.modes:
        coherence = "" if mode.coherence is None else f"{mode.coherence:.4f}"
        damping = "" if mode.damping_ratio is None else f"{mode.damping_ratio:.6f}"
        lines.append(
            ",".join(
                [
                    f"{mode.frequency_hz:.4f}",
                    "+".join(axis.upper() for axis in mode.axes),
                    mode.strongest_axis.upper(),
                    coherence,
                    damping,
                    "yes" if mode.damping_reliable else "no",
                    mode.quality,
                    f"{mode.prominence_db:.2f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _html(recording: Recording, analysis: Analysis, images: list[tuple[str, str]], csv_name: str) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    if analysis.warnings:
        warning_html = (
            "<div class='warn'><strong>Check these</strong><ul>"
            + "".join(f"<li>{escape(item)}</li>" for item in analysis.warnings)
            + "</ul></div>"
        )
    else:
        warning_html = "<p class='ok'>No measurement problems were flagged.</p>"
    rows = []
    for mode in analysis.modes:
        coherence = "—" if mode.coherence is None else f"{mode.coherence:.2f}"
        if mode.damping_reliable and mode.damping_ratio is not None:
            damping = f"{mode.damping_ratio * 100:.2f}%"
        else:
            damping = "not reliable"
        rows.append(
            "<tr>"
            f"<td>{mode.frequency_hz:.1f}</td>"
            f"<td>{escape(', '.join(axis.upper() for axis in mode.axes))}</td>"
            f"<td>{escape(mode.strongest_axis.upper())}</td>"
            f"<td>{coherence}</td>"
            f"<td>{damping}</td>"
            f"<td>{'Good' if mode.quality == 'good' else 'Check'}</td>"
            "</tr>"
        )
    if rows:
        table = (
            "<table><thead><tr>"
            "<th>Frequency (Hz)</th><th>Axes</th><th>Strongest axis</th>"
            "<th>Coherence</th><th>Approximate damping</th><th>Quality</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    else:
        table = "<p>No natural frequencies were found in the selected range.</p>"
    figures = []
    for caption, encoded in images:
        figures.append(
            "<figure>"
            f"<figcaption>{escape(caption)}</figcaption>"
            f"<img alt='{escape(caption)}' src='data:image/png;base64,{encoded}'>"
            "</figure>"
        )
    notes = "".join(f"<li>{escape(note)}</li>" for note in analysis.notes)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bracket natural frequencies — {escape(recording.source_name)}</title>
<style>
  body {{ font-family: "Segoe UI", Helvetica, Arial, sans-serif; color: #1c1c1c;
          background: #fff; max-width: 920px; margin: 32px auto; padding: 0 20px 48px;
          line-height: 1.45; }}
  h1 {{ font-size: 1.7rem; font-weight: 650; margin-bottom: 0.2rem; }}
  .meta {{ color: #555; margin-top: 0; }}
  .summary {{ font-size: 1.35rem; margin: 0.8rem 0 1rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.4rem 0 1.2rem; }}
  th {{ text-align: left; font-size: 0.82rem; color: #333; border-bottom: 2px solid #222; padding: 0.4rem; }}
  td {{ border-bottom: 1px solid #e4e4e4; padding: 0.45rem 0.4rem; }}
  .warn {{ background: #fff6e8; border-left: 4px solid #E69F00; padding: 0.2rem 0.9rem; }}
  .ok {{ color: #444; }}
  figure {{ margin: 1.5rem 0; break-inside: avoid; }}
  figcaption {{ font-weight: 600; margin-bottom: 0.35rem; }}
  img {{ width: 100%; height: auto; }}
  .notes {{ font-size: 0.9rem; color: #333; }}
  footer {{ font-size: 0.8rem; color: #666; margin-top: 2rem; }}
  @media print {{
    body {{ margin: 0; max-width: none; }}
    figure {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <h1>Bracket natural frequencies</h1>
  <p class="meta">{escape(recording.source_name)} · {escape(generated)}</p>
  <p class="summary">{escape(summary_sentence(analysis.modes))}</p>
  {warning_html}
  {table}
  {''.join(figures)}
  <h2>How to read this</h2>
  <p>{escape(_HOW_TO_READ)}</p>
  <h2>What was done</h2>
  <ul class="notes">{notes}</ul>
  <footer>Frequency table also saved as {escape(csv_name)}. Print this page from the browser if you want a PDF.</footer>
</body>
</html>
"""
