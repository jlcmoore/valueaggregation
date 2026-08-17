# Browser experiment

This directory contains the browser-based human experiment. The `mturk` name
is historical: the same task code was first used for MTurk and is now also run
through JATOS.

The current JATOS flow is:

1. `router.html` reads the conditions in `conditions.js` and sends the
   participant to the corresponding task component.
2. The task page loads its component settings and asks `jatos-assign.js` for a
   scenario row.
3. `main.js` builds the qualification and task questions from that row.
4. `graphs.js` draws the charts when the selected condition uses them.
5. `jatos-assign.js` submits the answers and updates the shared assignment
   queues.

## Main files

- `conditions.js`: Conditions enabled in the JATOS router and their component
  settings.
- `router.html`: Small JATOS component that assigns a condition and starts the
  matching task component.
- `health_single_with_qual.html`: Current combined qualification and experiment
  page.
- `template.html`: Pandoc wrapper for building a complete task page. It loads
  the CSS, JavaScript dependencies, JATOS support, and experiment scripts.
- `jatos-assign.js`: JATOS scenario assignment, stale-assignment recovery,
  submission, and quota handling.
- `main.js`: Loads scenario data, renders questions, checks qualification and
  attention items, and collects responses.
- `graphs.js`: D3 functions for the area and volume charts.
- `hitpub.css`: Shared task styling.
- `timeme.js`: Time-on-page tracking.
- `qualification_answers.json`: Answer key used by the qualification task and
  downstream analysis.
- `jatos_to_mturk.py`: Converts JATOS JSONL results to the wide, MTurk-style CSV
  format used by the analysis scripts.

## Other task versions

These files are retained because they document earlier versions of the study:

- `health.html`: Multi-scenario MTurk task template.
- `health_single.html`: Single-assignment MTurk task.
- `health_single_with_qual_no_volume.html`: Combined task using the earlier
  qualification without volume-chart questions.
- `health_no_charts.html`: Earlier no-chart task template.
- `qualification.html`: Standalone qualification task.
- `qualification_questions.html`: Qualification-question HTML fragment.
- `health-mturk.html`, `health_no_charts-mturk.html`, and
  `qualification-mturk.html`: Generated standalone MTurk pages.
- `variables.json`: Example/default experiment settings used by the older task
  setup. Current JATOS runs normally use component JSON instead.
- `example_command.sh`: Example of the older Pandoc/MTurk build command.

## Local viewers

- `view_scenario_file.html`: Loads scenarios from a CSV and displays their
  charts.
- `view_graph_from_json.html`: Displays charts for a pasted `GameState` JSON
  value.

To use the viewers, serve the intuition directory rather than opening the HTML
files directly:

```bash
cd experiments/intuition
python -m tools.simple_http_server
```

Then open the relevant page under `http://localhost:8000/mturk/`.
