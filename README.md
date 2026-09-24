# Bracket natural frequencies

A desktop tool for impact tests. You hit a metal bracket with an impulse hammer, measure the force and a triaxial accelerometer, and save the whole recording as one CSV. This opens that file, finds each blow, and reports the natural frequencies.

## Run it

From this folder:

```bash
python3 app.py
```

**Open example** loads a fake 10-hit recording so you can see the charts before a real file is ready. **Open CSV** loads your file. **Export report** writes an HTML page with the charts and a small CSV of the frequencies. Print the HTML page from the browser if you want a PDF.

The settings on the left update the charts. Leave the upper frequency blank to search up to 40% of the sample rate; the chart then zooms to the frequencies that were found. Type an upper frequency when you want the axis to stop at a number you chose.

## Your CSV

One file, many hammer blows, one triaxial accelerometer.

A typical header:

```text
time_s,force_N,accel_x_g,accel_y_g,accel_z_g
```

Units can also sit in parentheses, as in `Force (N)` and `Accel X (g)`. Volts are fine. The peak frequencies do not depend on calibration. Lines starting with `#`, and a short block of notes above the header, are skipped.

If the columns are only named `Channel 1`, `Channel 2`, and so on, the window asks you to assign time, force, X, Y, and Z. A file with no header and exactly five columns is treated as time, force, X, Y, Z, and that assumption is shown so you can change it. If there is no time column, type the sample rate.

## What the charts mean

- **Recording.** The whole force trace and the three acceleration traces, with each blow marked. A blow that was left out (a double-hit, or one too close to the next) is marked in a different color.
- **One blow.** The strongest accepted blow, with the part of each signal that is actually used shaded. The small inset is the hammer pulse.
- **Frequencies.** How strongly the bracket responded, per unit of hammer force. Bold lines are the average. Faint lines are the individual blows. The labels are the natural frequencies. The table under the chart lists which axes moved, how well the blows agreed (coherence), and an approximate damping.
- **Quality.** Coherence versus frequency, plus any problems: double-hits, clipping, blows that were too close, or a frequency range the hammer did not excite.

Coherence near 1 means the blows agreed. A peak that is not repeatable is not listed. For a lightly damped metal bracket, a peak in this acceleration-per-force curve is the natural frequency. The damping percentage comes from the width of the peak, with the fade of the analysis window removed. “Not reliable” means that width was not clean.

This is one sensor location. It does not draw a mode shape, and it is not a full curve-fit.

## The example file

`examples/bracket_10hits.csv` is simulated, not measured. It is one recording of ten hammer blows on a small bracket. The first natural frequency is near 100 Hz. A weak 60 Hz hum is in the acceleration and should not be reported as a natural frequency. The exact planted values used by the tests are in `examples/bracket_10hits_truth.json`.

Regenerate both files with:

```bash
python3 -m bracket_modes.synth
```

## Checks

```bash
python3 -m unittest discover -s tests -v
```
