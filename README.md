# Value Aggregation

This project looks at how to choose a collective action when people, groups, or moral views value the available outcomes differently.

The repository includes a Python package for aggregation rules and experiments
comparing those rules with human judgments, language-model judgments, and
choices in existing datasets. The main comparison is between rules such as
Nash bargaining and an inequality-efficiency (IE) family that trades off total
value against its distribution across groups.

## Paper

For details, see [Intuitions of Compromise: Utilitarianism vs.
Contractualism](https://arxiv.org/abs/2410.05496).

```bibtex
@misc{moore2024intuitions,
  title={Intuitions of Compromise: Utilitarianism vs. Contractualism},
  author={Jared Moore and Yejin Choi and Sydney Levine},
  year={2024},
  eprint={2410.05496},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2410.05496}
}
```

## What is in the repository

- `src/value_aggregation/` implements game states, normalization utilities,
  voting and bargaining rules, inequality and efficiency measures, and game
  generation helpers. `src/tests/` contains its unit tests.
- `experiments/intuition/` contains the main human and LLM compromise-judgment
  study: scenario generation, the browser task, model runners, statistical
  analyses, figures, and reproducibility instructions. Its
  [`README.md`](experiments/intuition/README.md) documents every subdirectory
  and the current paper workflow.
- `experiments/prevalence/` contains exploratory notebooks that translate the
  Kaleido, Moral Machine, and NLPositionality datasets into this project's
  common game representation.
- `demos.ipynb` demonstrates aggregation behavior on small moral-uncertainty
  examples.
- `external_data/` is the local destination for downloaded third-party data;
  it is not versioned. `make data` documents and downloads the expected
  sources.

## Setup

Python 3.11 or later is required. Create the canonical virtual environment and
install the package in editable mode with:

```bash
make init
source env-aggregation/bin/activate
```

The environment lives at `env-aggregation/`. Run the library test suite with:

```bash
make test
```

The intuition study additionally uses R with `brms` and `cmdstanr` for its
mixed-effects models and provider credentials for new LLM runs. See the study
README for those workflows.

## Running notebooks

With Jupyter installed in the active environment:

```bash
jupyter notebook
```

Most intuition-study commands expect the study directory as the working
directory:

```bash
cd experiments/intuition
```
