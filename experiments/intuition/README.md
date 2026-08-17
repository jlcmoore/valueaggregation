# Intuition Experiment

This directory contains the human and LLM compromise experiment. Participants
choose among proposals that distribute outcomes across groups of different
sizes. The scenario generator labels proposals using Nash bargaining and a
family of inequality-efficiency rules (IE, called `fehr` in some older code and
data). The analysis compares those labels with the observed choices and looks
at how the IE results change across values of alpha.

Run the commands below from `experiments/intuition/` with the repository virtual
environment active.

## Directory and code map

- `generation/scenario_utils.py` generates and validates scenario batches. It
  also contains older MTurk release and retrieval commands.
- `mturk/` is the browser experiment shared by MTurk and JATOS. It contains the
  task and qualification HTML, chart rendering, routing and assignment logic,
  styles, configuration examples, and `jatos_to_mturk.py`, which converts JATOS
  JSONL exports to the wide CSV shape expected by older analyses.
- `llm/new_model_experiment.py` is the batched LiteLLM runner.
- `analysis/` contains human and LLM descriptive tests, participant-level and
  qualification analyses, IE-family diagnostics, Pareto-dominance checks,
  preregistered classification tests, and the Python/R mixed-effects pipeline.
- `common/` contains scenario parsing, prompt construction, qualification, and
  dataframe helpers shared by generation, model runs, and analyses.
- `notebooks/` contains older exploratory analysis notebooks. The scripts in
  `analysis/` are the current workflow.
- `data/scenarios/` contains generated inputs; `data/results/llm/` contains
  model responses; and `data/analysis/` contains aggregate derived tables.
  These are eligible for public export.
- `data/results/jatos/` and `data/results/mturk/` contain raw or converted human
  response exports. Both may contain participant/platform identifiers and are
  excluded from public exports.
- `figures/` contains checked-in analysis outputs and source assets.
- `tools/` contains the small local HTTP server. `renv/` records the R
  environment bootstrap.
- `private/` contains cleanup notebooks, deployment notes, and the internal
  MTurk spending log. The whole directory is excluded from public exports.

## Prerequisites

- Python 3.11+
- R with `brms` and `cmdstanr` (for mixed-effects models)
- JATOS deployment configured for new human data collection (the internal
  deployment guide is intentionally private)
- API keys configured for LLM runs (for the providers/models you use)

## Canonical Scenario File

Most commands below assume this scenario file:

```bash
SCENARIO_FILENAME="data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=101_action-range=1,101_action-function-log=False_prevent-ties=True_agg-functions=['fehr','nash']_disagrees-only=False_num-scenarios=18_sample-size=150.csv"
```

The public export includes the scenarios and LLM results. Human analyses also
need the response CSV, which is excluded from the public export.

## Human Experiments (JATOS -> analysis)

### 1) Generate scenarios (18 trials per row)

```bash
python -m generation.scenario_utils scenarios \
  --num-agents 3 \
  --belief-steps 1 \
  --action-steps 101 \
  --number-scenarios 18 \
  --sample-size 150 \
  --max-actions 100 \
  --aggregation-functions fehr nash \
  --alpha-values 0 0.25 0.5 0.75 1 \
  --disagreements-per-hit 10 \
  --prevent-ties
```

### 2) Deploy scenarios to JATOS

Upload the scenario CSV as JATOS assets `data.csv`.

Authorized project members can consult
`private/operations/server_setup.md` for the deployment and export setup.

### 3) Export JATOS results and convert

```bash
python mturk/jatos_to_mturk.py \
  --input data/results/jatos/jatos_results_data_YYYYMMDDHHMMSS.txt \
  --scenario-file "$SCENARIO_FILENAME" \
  --desired-per-row 3
```

### 3b) Demographics representativeness checks

Set these paths to JATOS and demographics exports you are authorized to use:

```bash
JATOS_RESULTS="data/results/jatos/jatos_results_data_YYYYMMDDHHMMSS.txt"
DEMOGRAPHICS_FILE="data/results/jatos/prolific_demographic_export_YYYYMMDD.csv"

python mturk/jatos_to_mturk.py \
  --input "$JATOS_RESULTS" \
  --output /tmp/jatos_data_demographics_check.csv \
  --scenario-file "$SCENARIO_FILENAME" \
  --desired-per-row 3 \
  --demographics-file "$DEMOGRAPHICS_FILE"
```

To include failed qualification and attention checks, add
`--include-failed-qualification --include-failed-attention` to the command.

### 4) Run within-participant human analysis

```bash
python -m analysis.analysis_within_ppt \
  --num-scenarios 18 \
  --file "data/results/mturk/jatos_data_condition=best compromise-area.csv" \
  --qual-type area \
  --attention-threshold 1.0 \
  --auto-classification-threshold
```

For all ppts regardless of qual or attention check:

```bash
python -m analysis.analysis_within_ppt \
  --num-scenarios 18 \
  --file "data/results/mturk/jatos_data_condition=best compromise-area.csv" \
  --qual-type none \
  --attention-threshold 0 \
  --auto-classification-threshold
```

Preregistered exploratory attention-threshold robustness summary (for example,
the 75% attention-pass appendix analysis):

```bash
python -m analysis.analysis_within_ppt \
  --num-scenarios 18 \
  --file "data/results/mturk/jatos_data_condition=best compromise-area.csv" \
  --qual-type area \
  --auto-classification-threshold \
  --attention-thresholds 0.75
```


Primary outputs:
- `figures/new_data_classification_by_alpha.pdf`
- `figures/new_data_agree_classification_by_alpha.pdf`

### 5) Run the IE-family diagnostic robustness analysis

This analysis restricts the human data to trials where the Nash proposal is
not optimal under the Inequality Sum for any alpha value in `[0, 1]`. It
compares Nash with common-alpha and hierarchical personal-alpha IE models using
participant-marginal BIC and participant-cross-fitted held-out prediction.

```bash
python -m analysis.analysis_family_diagnostic \
  --file "data/results/mturk/jatos_data_condition=best compromise-area.csv" \
  --qualification-answers mturk/qualification_answers.json \
  --num-scenarios 18 \
  --attention-threshold 1.0 \
  --folds 5 \
  --repeats 10 \
  --output-json data/analysis/family_diagnostic_human.json
```

### 6) Audit Pareto dominance and rerun the human tests by stratum

This analysis counts whether the IE-selected proposal is Pareto dominated in
each unique disagreement scenario. It then repeats the participant-level
Nash-versus-IE Wilcoxon test separately for dominated and undominated IE
proposals at each alpha level.

```bash
python -m analysis.analysis_pareto_dominance \
  --output-directory data/analysis
```

Primary outputs:
- `data/analysis/pareto_dominance_scenario_counts.csv`
- `data/analysis/pareto_dominance_choice_tests.csv`

### 7) Generate the main example figure

The checked scenario is an alpha = 0.5 disagreement in which the IE-selected
proposal is Pareto undominated. The chart itself comes from the experiment's
`mturk/graphs.js::make_area_chart` function.

```bash
python -m http.server 8000
```

Open
`http://localhost:8000/mturk/generate_main_scenario_figure.html`, download the
generated SVG, and replace `figures/scenario_borders.svg`. Then export the PDF:

```bash
inkscape figures/scenario_borders.svg \
  --export-type=pdf \
  --export-filename=figures/scenario_borders.pdf
```


## LLM Experiments

### 1) Run main condition files

Qwen, area:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model qwen-3.5-397b-a17b-no-reasoning \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16 \
  --conditions area
```

Qwen, none:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model qwen-3.5-397b-a17b-no-reasoning \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16 \
  --conditions none
```

GPT-5.5, area:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model gpt-5.5-no-reasoning \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16 \
  --conditions area
```

GPT-5.5, none:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model gpt-5.5-no-reasoning \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16 \
  --conditions none
```

### 1b) Full high-reasoning runs (area only, no qualification)

These commands run the full scenario file (`sample-size=150`, `num-scenarios=18`)
for the `area` condition only. They do not run qualification.

Qwen high reasoning, area:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model qwen-3.5-397b-a17b-high-reasoning \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16 \
  --max-tokens 4096 \
  --conditions area
```

GPT-5.5 high reasoning, area:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model gpt-5.5-high-reasoning \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16 \
  --max-tokens 256 \
  --conditions area
```

### 2) Run qualification task (area)

Qwen:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model qwen-3.5-397b-a17b-no-reasoning \
  --qualification \
  --qualification-sampling row \
  --chart-type area \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16
```

GPT-5.5:

```bash
python -m llm.new_model_experiment "$SCENARIO_FILENAME" \
  --model gpt-5.5-no-reasoning \
  --qualification \
  --qualification-sampling row \
  --chart-type area \
  --samples 1 \
  --batch-size 16 \
  --max-workers 16
```

### 3) Analyze LLM outputs

no reasoning

```bash
RUN_BASE="$(basename "$SCENARIO_FILENAME" .csv)"

QWEN_DIR="data/results/llm/Qwen_Qwen3.5-397B-A17B-no-reasoning/$RUN_BASE"
GPT55_DIR="data/results/llm/gpt-5.5-no-reasoning/$RUN_BASE"

python -m analysis.analysis_within_llm \
  --file "$QWEN_DIR/temp=default_api=chat_qualification=False_show-charts=True_chart-type=area_samples=1_zero-shot=False.csv" \
  --output-prefix qwen_area

python -m analysis.analysis_within_llm \
  --file "$QWEN_DIR/temp=default_api=chat_qualification=False_show-charts=False_chart-type=area_samples=1_zero-shot=False.csv" \
  --output-prefix qwen_none

python -m analysis.analysis_within_llm \
  --file "$GPT55_DIR/temp=default_api=chat_qualification=False_show-charts=True_chart-type=area_samples=1_zero-shot=False.csv" \
  --output-prefix gpt55_area

python -m analysis.analysis_within_llm \
  --file "$GPT55_DIR/temp=default_api=chat_qualification=False_show-charts=False_chart-type=area_samples=1_zero-shot=False.csv" \
  --output-prefix gpt55_none
```

reasoning

```bash
RUN_BASE="$(basename "$SCENARIO_FILENAME" .csv)"

QWEN_DIR="data/results/llm/Qwen_Qwen3.5-397B-A17B-high-reasoning/$RUN_BASE"
GPT55_DIR="data/results/llm/gpt-5.5-high-reasoning/$RUN_BASE"

python -m analysis.analysis_within_llm \
  --file "$QWEN_DIR/temp=default_api=chat_qualification=False_show-charts=True_chart-type=area_samples=1_zero-shot=False.csv" \
  --output-prefix qwen_area_reasoning

python -m analysis.analysis_within_llm \
  --file "$GPT55_DIR/temp=default_api=chat_qualification=False_show-charts=True_chart-type=area_samples=1_zero-shot=False.csv" \
  --output-prefix gpt55_area_reasoning

```

### 4) Analyze LLM qualification outputs

```bash
python -m analysis.analysis_qualification_llm \
  --file "$QWEN_DIR/temp=default_api=chat_qualification=True_show-charts=False_chart-type=area_samples=1_qual-sampling=row_zero-shot=False.csv" \
  --chart-type area

python -m analysis.analysis_qualification_llm \
  --file "$GPT55_DIR/temp=default_api=chat_qualification=True_show-charts=False_chart-type=area_samples=1_qual-sampling=row_zero-shot=False.csv" \
  --chart-type area
```

## Notes

- `--conditions area` maps to `show-charts=True` in output filenames.
- `--conditions none` maps to `show-charts=False`.
- Use `--max-prompts` only for smoke tests.
- For Together retry issues, lower concurrency (for example `--batch-size 1 --max-workers 1`).
