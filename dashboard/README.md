# dashboard/ — live spectrum waterfall

Visual console for the simulator: a 50-channel waterfall with the receiver's
scan track drawn over the hidden ground truth, plus running and whole-mission
metrics. Specified in the RF build doc §5 and the ML build doc §16 item 17.

## Run it

```
python -m dashboard.build
```

That re-exports fresh mission data, rebuilds `index.html` and opens it in your
browser. The page is a single self-contained file, so it also works by
double-clicking `index.html` or sharing the file.

```
python -m dashboard.build --no-open      # build only
python -m dashboard.build --no-refresh   # reuse the existing runs.json
python -m dashboard.export_runs          # regenerate runs.json alone
```

## What you see

Rows are channels, columns are time slots. Grey marks are moments a transmitter
was genuinely active; the scan track shows where the receiver listened — teal
for an intercept, amber for a blind slot after retuning, red for a real signal
it missed, faint grey for listening to an empty channel. Switch strategies with
the buttons, scrub to any slot, or change the replay speed.

## Files

| File | Purpose |
|---|---|
| `export_runs.py` | Runs every built-in strategy and writes `runs.json` |
| `template.html` | The page, with a `__RUNS__` placeholder for the data |
| `build.py` | Inlines the data into `index.html` and opens it |
| `runs.json`, `index.html` | Generated; safe to delete and rebuild |

Everything drawn comes from a real run of this simulator on
`scenarios/train_default.json`, seed 0. The animation covers the first 600 of
5,000 slots; the full-mission figures cover all 5,000.

A live Streamlit version that re-runs the Python on each interaction is the
scheduler team's option; this static build was chosen so the console opens
anywhere, including a phone, with nothing installed.
