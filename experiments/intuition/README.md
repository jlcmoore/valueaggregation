
These are the files used to run the main experiments in our paper on both human subjects on Mturk and LLMs.

## Structure

- `data`: Generated and results data.
- `mturk`: Files for use in making mturk HITs
- `analysis.ipynb`: Various analyses of the output llm and mturk experiments.
- `experiment.yaml`: A Beaker experiment specification for running llm experiments with  `model_experiment.py`.
- `model_experiment.py`: A utility for surveying llms (openai, anthropic, or llm-lab currently supported). 
- `scenario_utils.py`: A utility for generating games for HITs for mturk and llm experiment, processing the results of mturk experiments, and splitting up files across HITs.
- `shared_analysis.py`: Various analysis code, mostly in `pandas` for anlyzing the `data/results`. Used in `analysis.ipynb`
- `simple-http-servery.py`: A server to allow local files (such as referenced by `mturk/view_scenario_file.html`) to be opened by javascript in a browser.
- `utils.py`: Utility functions shared between files.

### `data`

- `results`
    - `llm`: The output of various language models from `model_experiment.py`
        - `claude-2`
            `maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0/`
                - The files resulting from running all of the relevant conditions on each LLM.
                - `temp=1_api=completion_qualification=False_show-charts=False_chart-type=area_samples=10_zero-shot=False.csv`
                - `temp=1_api=completion_qualification=False_show-charts=True_chart-type=area_samples=10_zero-shot=False.csv`
                - `temp=1_api=completion_qualification=False_show-charts=True_chart-type=both_samples=10_zero-shot=False.csv`
                - `temp=1_api=completion_qualification=False_show-charts=True_chart-type=volume_samples=10_zero-shot=False.csv`
                - `temp=1_api=completion_qualification=True_show-charts=False_chart-type=area_samples=10_zero-shot=False.csv`
                - `temp=1_api=completion_qualification=True_show-charts=False_chart-type=volume_samples=10_zero-shot=False.csv`
        - `claude-3-opus-20240229`                
        - `davinci-002`
        - `gpt-3.5-turbo-16k`
        - `gpt-4`
    - `mturk`: The output of various mturk HITs. Included are the three final results files.
        -  `maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0/`
            - `chart_type=area_maximize=True.csv`
            - `chart_type=both_maximize=True.csv`
            - `chart_type=volume_maximize=True.csv`
            - `chart_type=none_maximize=True.csv`
- `scenarios`: Generated GameStates in `json` format output as `csv`s for use in mturk and llm experiments.
    - `maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv`
        - This is the "Focused" set of scenarios
    - `maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv`
        - This is the "Random" set of scenarios

### `mturk`

- `graph.js`: Javascript utilities for drawing stacked area charts (MEC) and 3d volume charts (NBS).
- `health.html`: The template for a HIT involving health-based scenarios for use in assigning an entire `csv` file from `../data/scenarios`.
- `health_single.html`: The template for a HIT involving health-based scenarios for use in a single assignment. Used in conjunction with `../scenario_utils.py`.
- `health_single_with_qual.html`: Like `health_single.html` but contains the qualification task as a component.
- `hitpub.css`: The style file for the HITs.
- `main.js`: The main program for the HITs. Generates the questions and charts programmatically.
- `qualification.html`: A HIT to qualify workers for the proper experiment for use with a single (non data based) assignment. Contains a number of questions which can be validated with `../scenario_utils.py`
- `qualification_answers.json`: A data file containing the answers to the above qualification task.
- `template.html`: The template file by which `pandoc` generates the collated html for the HITs. Contains headers, footers, and includes.
- `timeme.js`: Timing code for the HITs.
- `variables.json`: Experimental variables for the HITs. `"chart_type" \in ["both", "area", "volume"]` `"maximize" \in [true, false]`.
- `view_scenario_file.html`: For use in viewing the volume and area graphs of a `csv` file from `../data/`. Because of security issues, you must run `cd experiments/intuition;  python simple-https-server.py` in order for it to work.
- `view_graph_from_json.html`: For use in viewing the volume and area graphs of a `GameState` input as `json`.

## Running Mturk

You'll have to install `pandoc` to generate the files for <https://requester.mturk.com/>. 
E.g.:

`brew install pandoc`

For example to generate the qualification file run this command:

`pandoc --standalone -f html+raw_html -t html --standalone --embed-resources --template template.html -o qualification-mturk.html qualification.html`

To generate a single health survey HIT run this command. Note that the file will be empty of graphs because those are passed in as arguments processed by the back-end webserver of mturk or by pandoc locally (e.g. variables are indicated like so `${scenario_1_json}`).

`pandoc --standalone -f html+raw_html -t html --standalone --embed-resources --template template.html -o health-mturk.html health.html`

### Final

This is the command we called to generate the final scenarios.

```
python scenario_utils.py scenarios --num-agents 3 --belief-steps 1 --action-steps 3  --number-scenarios 4 --sample-size 34
```

yielding

```
INFO:value_aggregation.utils:number of scenarios: 19657
INFO:value_aggregation.utils:number of disagreements: 162
INFO:value_aggregation.utils:percent disagreements: 0.0082
```

(
Add `--aggregation-functions nash fehr util` to compare between all three of Nash Product, Utilitarian Sum, and Inequality Sum.

```
python scenario_utils.py scenarios --num-agents 3 --belief-steps 1 --action-steps 101 --number-scenarios 4 --sample-size 34 --max-actions 100 --aggregation-functions nash fehr util --prevent-ties
```

```
INFO:value_aggregation.utils:number of scenarios: 999901
INFO:value_aggregation.utils:number of disagreements: 2358
INFO:value_aggregation.utils:percent disagreements: 0.0030

INFO:value_aggregation.utils:number of scenarios: 999901
INFO:value_aggregation.utils:number of disagreements: 2292
INFO:value_aggregation.utils:percent disagreements: 0.0029

INFO:value_aggregation.utils:number of scenarios: 999901
INFO:value_aggregation.utils:number of disagreements: 1728
INFO:value_aggregation.utils:percent disagreements: 0.0021
```
)

(And this for comparing with the Inequality Sum + changing group size

```
python scenario_utils.py scenarios --num-agents 3 --belief-range .1 .9 --belief-steps 10 --action-steps 101 --number-scenarios 4 --sample-size 34 --max-actions 100 --aggregation-functions nash fehr util --prevent-ties --greedy-generation
```

)

(And this for 'random' actions.

```
python scenario_utils.py scenarios --num-agents 3 --belief-steps 1 --action-steps 101 --number-scenarios 4 --sample-size 34 --max-actions 100
```

running this three times yields: 172144 / 999901, 17.2%

```
INFO:value_aggregation.utils:number of scenarios: 999901
INFO:value_aggregation.utils:number of disagreements: 166554
INFO:value_aggregation.utils:percent disagreements: .1666

INFO:value_aggregation.utils:number of scenarios: 999901
INFO:value_aggregation.utils:number of disagreements: 174306
INFO:value_aggregation.utils:percent disagreements: 0.1743

INFO:value_aggregation.utils:number of scenarios: 999901
INFO:value_aggregation.utils:number of disagreements: 175572
INFO:value_aggregation.utils:percent disagreements: 0.1756
```
)

(And this to compare with the logarithmic outcomes

```
python scenario_utils.py scenarios --num-agents 3 --belief-steps 1 --action-steps 3  --number-scenarios 4 --sample-size 34  --action-function-logarithmic
```

```
INFO:value_aggregation.utils:number of scenarios: 19657
INFO:value_aggregation.utils:number of disagreements: 486
INFO:value_aggregation.utils:percent disagreements: 0.0247
```
)

And this is how we released the studies for three conditions and got the results for each condition.

Volume: 

```
python scenario_utils.py release --filename data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --chart-type volume
```

```
python scenario_utils.py results --filename data/results/mturk/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0/chart_type=volume_maximize=True.csv
```

Area

```
python scenario_utils.py release --filename data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --chart-type area
```

```
python scenario_utils.py results --filename data/results/mturk/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0/chart_type=area_maximize=True.csv
```

Both

```
python scenario_utils.py release --filename data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --chart-type both
```

```
python scenario_utils.py results --filename data/results/mturk/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0/chart_type=both_maximize=True.csv
```

Note that we did have to run qualification tasks using `mturk/qualification.html` to get qualified workers. We also had to run the above commands many times as we released each of 102 set of four questions to one unique worker, many of whom never complete the survey. Many of those workers also answer inconsistently. See `delete_inconsistent_workers.ipynb` for filtering those out.

### (LL-)Model Experiments

Here are listed the commands to run our experiments on the `claude-2.1` model, assuming your key is stored in `$ANTHROPIC_API_KEY`. To run on openai models make sure your key is stored in `$OPENAI_API_KEY`.

The two qualificaiton conditions.
```
python experiments/intuition/model_experiment.py --output_directory /experiments/intuition/data/results --samples 10 --temperature 1 --endpoint completion --model claude-2.1 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --qualification --chart-type area

python experiments/intuition/model_experiment.py --output_directory /experiments/intuition/data/results --samples 10 --temperature 1 --endpoint completion --model claude-2.1 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --qualification --chart-type volume

\

python experiments/intuition/model_experiment.py --output_directory experiments/intuition/data/results --samples 10 --temperature 1 --endpoint chat --model claude-3-opus-20240229 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --qualification --chart-type area

python experiments/intuition/model_experiment.py --output_directory experiments/intuition/data/results --samples 10 --temperature 1 --endpoint chat --model claude-3-opus-20240229 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --qualification --chart-type volume

python experiments/intuition/model_experiment.py --output_directory experiments/intuition/data/results --samples 10 --temperature 1 --endpoint chat --model claude-3-opus-20240229 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=101_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --qualification --chart-type area

python experiments/intuition/model_experiment.py --output_directory experiments/intuition/data/results --samples 10 --temperature 1 --endpoint chat --model claude-3-opus-20240229 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=101_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --qualification --chart-type volume

python experiments/intuition/model_experiment.py --output_directory experiments/intuition/data/results --samples 10 --temperature 1 --endpoint chat --model claude-3-opus-20240229 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=101_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --qualification --chart-type area
```


Each of the area, volume, both and none conditions.
```
python experiments/intuition/model_experiment.py --output_directory /experiments/intuition/data/results --samples 10 --temperature 1 --endpoint completion --model claude-2.1 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --show-chart --chart-type area

python experiments/intuition/model_experiment.py --output_directory /experiments/intuition/data/results --samples 10 --temperature 1 --endpoint completion --model claude-2.1 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --show-chart --chart-type volume

python experiments/intuition/model_experiment.py --output_directory /experiments/intuition/data/results --samples 10 --temperature 1 --endpoint completion --model claude-2.1 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic --show-chart --chart-type both

python experiments/intuition/model_experiment.py --output_directory /experiments/intuition/data/results --samples 10 --temperature 1 --endpoint completion --model claude-2.1 experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=3_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv --anthropic
```

For other models subsitute `--model claude-2.1 --endpoint completion --anthropic` with one of the following:

- `--endpoint chat --model claude-3-opus-20240229 --anthropic`
- `--endpoint completion --model davinci-002 --openai`
- `--endpoint chat --model gpt-3.5-turbo-16k-0613 --openai`
- `--endpoint chat --model gpt-4-0613 --openai`

Also replace the scenario file with `experiments/intuition/data/scenarios/maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_action-steps=101_action-range=1,101_action-function-log=False_num-scenarios=4_sample-size=68.0.csv` for the "random" scenarios as well.
