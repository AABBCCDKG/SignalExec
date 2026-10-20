# SignalExec dashboard

Build-free static results page using local HTML, CSS, JavaScript, JSON, and SVG.

## Run

From the repository root:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/dashboard/>.

The public deployment is <https://aabbccddkg.github.io/SignalExec/>.

## Update results

Edit `dashboard/data/experiments.json` only after an interval's formal run is
complete. Keep unfinished intervals at `chartData: null`; the page will show an
empty state and cannot borrow another interval's values.

Current state:

- `1m`, `3m`, `5m`, `15m`, `30m`: five-symbol locked comparison complete and
  published for May-August 2026.

The earlier `5m` single-BTC short-history prototype is research history, not a
dashboard result. Publish an interval only after the same five-symbol date range
has completed; refresh the underlying monthly archives before future reruns.

`scripts/export_interval_dashboard.py` regenerates every `signalResponse` and
`vsTwap` payload from merged locked-test artifacts. Method `mean` and `seeds`
remain separate. The SVG paths animate when they enter the viewport;
`prefers-reduced-motion` renders the final state immediately.

Research data stays local. Public reproduction uses official Binance URLs plus
the repository's SHA256 manifest/downloader. Raw archives, caches, and models
are not hosted or redistributed by this dashboard.

## Checks

```bash
jq empty dashboard/data/experiments.json
node --check dashboard/app.js
python3 -m http.server 8000
curl --fail --silent http://localhost:8000/dashboard/ >/dev/null
curl --fail --silent http://localhost:8000/dashboard/data/experiments.json | jq empty
```
