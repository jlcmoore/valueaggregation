'''
Various utilities, mostly in pandas, for reading in the output of the mturk 
and llm experiments, processing, and analyzing the resulting data.

Author: Jared Moore
'''

import itertools
import logging
import os
import pandas as pd
import scipy.stats
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
import jinja2
import json


logging.basicConfig(level=logging.INFO)

import model_experiment
import value_aggregation as pm
from utils import QUESTION_STRINGS, PROBABILITY_COLUMNS

pd.options.mode.copy_on_write = True

MODEL_NAMES_SHORT = {
    'gpt-4' : 'gpt-4',
    'gpt-3.5-turbo-16k' : 'gpt-3.5',
    'davinci-002' : 'davinci-002',
    'claude-2' : 'claude-2',
    'claude-3-opus-20240229' : 'claude-3', #-- once qualification is complete
}

MTURK_CONDITIONS_SHORT = {
    "chart_type=area_maximize=True.csv" : "Area",
    "chart_type=volume_maximize=True.csv" : "Volume",
    "chart_type=both_maximize=True.csv" : "Both",
    "chart_type=none_maximize=True.csv" : "None"
}

MODELS = list(MODEL_NAMES_SHORT.values())
MODELS_LONG = list(MODEL_NAMES_SHORT.keys())

nbs_long = "Nash Product"
mec_long = "Utilitarian Sum"
nbs_short = "∏"
mec_short = "∑"


FUNCTION_NAMES_TO_SHORT = {
'scenario_mec' : mec_short,
'scenario_nbs' : nbs_short,
'scenario_fehr' : '=',
'scenario_rawls' : '<', 
}

_COLORS = [colormaps['Set2'](x) for x in range(10)]
AGGREGATION_TO_COLOR = {agg : color for agg, color in zip(FUNCTION_NAMES_TO_SHORT.keys(), _COLORS)}

CONDITION_ORDER = ['Area', 'Volume', 'Both', 'None']

CONDITION_TO_COLOR = {condition : color for condition, color in 
                      zip(CONDITION_ORDER, _COLORS[len(AGGREGATION_TO_COLOR):])}

INEQUALITY_SUPPORT = 3/10

## Static Variables

SCENARIOS_DIR = 'data/scenarios'
MTURK_RESULTS_DIR = "data/results/mturk"
LLM_RESULTS_DIR = 'data/results/llm'


num_scenarios = 4

LLM_TEST_PLOT_WIDTH = .95
BOX_PLOT_WIDTH = .7
PERCENT_PLT_YLIM = (-0.05,1.05)


def sort_by_condition(dfs):
    return sorted(dfs, key=lambda x: CONDITION_ORDER.index(x.name))

def shorter_llm_condition(condition):
    name = ""
    if "qualification=True" in condition:
        if 'volume' in condition:
            return 'q. Volume'
        else:
            return 'q. Area'
    else:
        if "both" in condition:
            return "Both"
        elif "show-charts=False" in condition:
            return "None"
        elif 'volume' in condition:
            return "Volume"        
        else:
            return "Area"


def explode_df(df, num_scenarios):
    variable_col_names = {'scenario_{0}_hash' : 'scenario_hash',
                          'scenario_{0}_json' : 'scenario_json',
                          'scenario_{0}_nbs' : 'scenario_nbs',
                          'scenario_{0}_mec' : 'scenario_mec',
                          'scenario_{0}_mft' : 'scenario_mft'}
    frames = []
    for i in range(1, num_scenarios + 1):
        num_cols = {k.format(i): v for k, v in variable_col_names.items()}
        selected = df[list(num_cols.keys())].rename(columns=num_cols)
        selected['question'] = i
        frames.append(selected)
    return pd.concat(frames)

def input_dtypes(num_scenarios):
    cols = {'Input.scenario_{0}_hash' : np.int64}
    res = {}
    for i in range (1, num_scenarios + 1):
        for k, v in cols.items():
            res[k.format(i)] = v
    return res

def mturk_explode_df(df, num_scenarios):
    keep_columns = ['WorkerId']
    variable_columns = {'Input.scenario_{0}_hash' : 'scenario_hash',
                        'Input.scenario_{0}_json' : 'scenario_json',
                        'Input.scenario_{0}_nbs' : 'scenario_nbs',
                        'Input.scenario_{0}_mec' : 'scenario_mec',
                        'Input.scenario_{0}_mft' : 'scenario_mft',
                        'Answer.q_question-{0}' : 'response',
                        'Answer.q_question-{0}_attn' : 'attention-response',
                        'Answer.q_question-{0}_attn_answer' : 'attention-answer'}

    # The first step is to flatten the data so each question has its own row
    frames = []
    for i in range(1, num_scenarios + 1):
        num_cols = {k.format(i): v for k, v in variable_columns.items()}
        selected = df[keep_columns + list(num_cols.keys())].rename(columns=num_cols)
        selected['question'] = i
        frames.append(selected)
    df_exploded = pd.concat(frames)
    return df_exploded


def control(df, between=('scenario_nbs', 'scenario_mec')):
    return df[is_control(df, between)]

def is_control(df, between=('scenario_nbs', 'scenario_mec')):
    first = between[0]
    second = between[1]
    return ((df[first] == df[second]) 
            | df[first].isnull() | df[second].isnull())

def workers_pass_control(df):
    cdf  = control(df)
    workers_pass = cdf[cdf['scenario_mec'] == cdf['response']]['WorkerId'].unique()
    return df[df['WorkerId'].isin(workers_pass)]

def uncertain(df):
    return df[(df['scenario_nbs'] == df['scenario_mec']) 
            & df['scenario_mec'].isnull() | df['scenario_nbs'].isnull()]

def test(df, between=('scenario_nbs', 'scenario_mec')):
    # Both are not equal and both are not null
    first = between[0]
    second = between[1]
    return df[(df[first] != df[second]) &
            df[first].notnull() & df[second].notnull()]

def fair_nbs_test(df):
    # Both are not equal and both are not null
    return df[(df['scenario_fair'] != df['scenario_nbs']) &
            df['scenario_fair'].notnull() & df['scenario_nbs'].notnull()]

def fair_nbs_control(df):
    # Both are not equal and both are not null
    return df[(df['scenario_fair'] == df['scenario_nbs'])
            | df['scenario_fair'].isnull() | df['scenario_nbs'].isnull()]

def fair_mec_test(df):
    # Both are not equal and both are not null
    return df[(df['scenario_fair'] != df['scenario_mec']) &
            df['scenario_fair'].notnull() & df['scenario_mec'].notnull()]

def fair_mec_control(df):
    # Both are not equal and both are not null
    return df[(df['scenario_fair'] == df['scenario_mec'])
            | df['scenario_fair'].isnull() | df['scenario_mec'].isnull()]

def fair_mec_nbs_test(df):
    # Both are not equal and both are not null
    return df[((df['scenario_fair'] != df['scenario_mec']) &
            df['scenario_fair'].notnull() & df['scenario_mec'].notnull()) &
    ((df['scenario_fair'] != df['scenario_nbs']) &
            df['scenario_fair'].notnull() & df['scenario_nbs'].notnull())]

def fair_mec_nbs_control(df):
    # Both are not equal and both are not null
    return df[((df['scenario_fair'] == df['scenario_mec'])
            | df['scenario_fair'].isnull() | df['scenario_mec'].isnull()) &
    ((df['scenario_fair'] == df['scenario_nbs'])
            | df['scenario_fair'].isnull() | df['scenario_nbs'].isnull())]

def fair_test_mec_nbs_control(df):
    return control(fair_mec_nbs_test(df))

def get_intersection(one, other):
    return one[one['scenario_hash'].isin(other['scenario_hash'])]

def when_not_both_nan(df):
     return df[(df['scenario_mec'] == df['scenario_nbs']) |
                    (df['scenario_mec'].notnull()) |
                    (df['scenario_nbs'].notnull())]

def add_response_distribution(df):
    '''
    For the input methods that do not return a disribution by default. 
    Necessary for mturk and the chat endpoint for models
    '''
    counts = df['response'].value_counts()
    counts = counts.div(counts.sum())
    
    for question in QUESTION_STRINGS:
        if question not in counts:
            counts[question] = np.nan

    response = counts.index[counts.argmax()]
    counts = counts.reindex(index=QUESTION_STRINGS)
    return pd.DataFrame([tuple(counts) + (response,)], columns=PROBABILITY_COLUMNS + ['response'])

def get_response_dist(df, columns):
    if set(PROBABILITY_COLUMNS).issubset(set(df.columns)):
        function = aggregate_response_distribution
    else: 
        function = add_response_distribution
    dist = df.groupby(columns).apply(function).droplevel(level=len(columns))
    cols = columns + ['scenario_json','scenario_mec', 'scenario_nbs', 'scenario_mft']
    right = df[cols].drop_duplicates()
    dist = dist.merge(right, on=columns, how='left')
    return dist

def aggregate_response_distribution(df):
    '''
    For the input methods starting with `PROBABILITY_COLUMNS`
    '''
    result = ()
    max_sum = None
    max_ans = None
    for answer, column in zip(QUESTION_STRINGS, PROBABILITY_COLUMNS):
        new_sum = df[column].sum()/ len(df)
        if max_sum is None or new_sum > max_sum:
            max_ans = answer
            max_sum = new_sum
        result += (df[column].sum()/ len(df),)

    return pd.DataFrame([result + (max_ans,)], columns=PROBABILITY_COLUMNS + ['response'])

# Example usage:
# get_response_dist(mturk_df, ['scenario_hash', 'question'], add_response_distribution)
# get_response_dist(mturk_df, ['scenario_hash'], add_response_distribution)
# get_response_dist(llm_df, ['scenario_hash'], aggregate_response_distribution)

def apply_run_game(row, run_func, **kwargs):
    gs = pm.decode_gameState(row['scenario_json'])
    answer = run_func(gs, none_if_tie=True, **kwargs)
    return answer

def add_function(df, function, name, **kwargs):
    result = df.apply(apply_run_game, axis=1, run_func=pm.run_equality_efficiency, **kwargs)
    df[f'scenario_{name}'] = result
    return df

def add_fehr(df, **kwargs):
    return add_function(df, pm.run_equality_efficiency, 'fehr', **kwargs)

def add_mft(df):
    return add_function(df, pm.run_mft, 'mft')

def add_fair(df):
    return add_function(df, pm.run_fair_dominated, 'fair')

def add_rawls(df):
    return add_function(df, pm.run_fair_dominated, 'rawls')    

def mturk_preprocess_df(df):
    df_exploded = mturk_explode_df(df)
    # only look at those which pass the attention checks
    return (df_exploded[df_exploded['attention-answer'] == 
                df_exploded['attention-response']])

def llm_preprocess_df(df):
    return df[df['response'].isin(QUESTION_STRINGS)]

# Answering: what percent of the time do they get it right? 

def get_answer_percent(row):
    return row[f"probability: {row['response']}"]

def get_mec_percent(row):
    if row['scenario_mec'] is np.NaN:
        return np.NaN
    return row[f"probability: {row['scenario_mec']}"]

def get_mft_percent(row):
    if row['scenario_mft'] is np.NaN:
        return np.NaN
    return row[f"probability: {row['scenario_mft']}"]

def get_nbs_percent(row):
    if row['scenario_nbs'] is np.NaN:
        return np.NaN
    return row[f"probability: {row['scenario_nbs']}"]

def get_mec_rank(row):
    dist = row[PROBABILITY_COLUMNS].sort_values(ascending=False)
    model_ans = row['scenario_mec']
    if model_ans is np.NaN:
        return np.NaN
    idx = dist.index.get_loc(f'probability: {model_ans}')
    if dist[idx] is np.NaN:
        return np.NaN
    return idx + 1

def null_ties(df):
    return df[df['scenario_mec'].isnull() & df['scenario_nbs'].isnull()]

def not_null_ties(df):
    condition = [True] * len(df)
    for function in FUNCTION_NAMES_TO_SHORT.keys():
        if function in df:
            condition &= ~df[function].isnull()
    return df[condition]

def not_null_ties_fair(df):
    return df[~df['scenario_fair'].isnull()]

## How much support did the max answer get? (Degree of unanimity)

def plot_answer_distribution(df, **kwargs):
    
    name = None
    if hasattr(df, 'name'):
        name = df.name
        kwargs['title'] = name

    ties = null_ties(df)
    df = control(df)
    max_answer = df.apply(get_answer_percent, axis=1)
    max_answer.name = 'mx'

    right_answer = df.apply(get_mec_percent, axis=1)
    right_answer.name = 'right'
    
    max_answer_tie = ties.apply(get_answer_percent, axis=1)
    max_answer_tie.name = 'mx-ties'
    
    result = []
    if len(max_answer) > 0:
        result.append(max_answer.to_frame())

    if len(max_answer_tie) > 0:
        result.append(max_answer_tie.to_frame())
    
    if len(right_answer) > 0:
        result.append(right_answer.to_frame())

    pd.concat(result).plot.box(**kwargs)

    # TODO: draw dotted line at 50%?

def plot_rank_answer_distribution(df):
    series = []
    dist = df
    for rank in range(1, len(QUESTION_STRINGS) + 1):
        mec_ans = dist[dist['mec rank'] == rank].apply(get_mec_percent, axis=1)
        mec_ans.name = f'mec r. {rank}'
        
        if len(mec_ans) > 0:
            mec_ans = mec_ans.to_frame()
            series.append(mec_ans)

    pd.concat(series).plot.box()


def top_in_answers(df):
    '''
    Does it agree with either NBS or MEC?
    '''
    mec_nbs = [df['scenario_mec'].unique()[0], df['scenario_nbs'].unique()[0]]
    counts = df['response'].value_counts()
    counts = counts.div(counts.sum())
    # index of max
    return counts.idxmax() in mec_nbs

def model_agree_stats(d, percentage=False, fair=False):
    d = not_null_ties(d)
    if percentage:
        demoninator = len(d)
    else:
        demoninator = 1

    values =(len(d[d['scenario_nbs'] == d['response']]) / demoninator,
                      len(d[d['scenario_mec'] == d['response']]) / demoninator)
    columns = ['nbs', 'mec']

    if fair and 'scenario_fair' in d:
        values += (len(d[d['scenario_fair'] == d['response']]) / demoninator,)
        columns += ['fair']

    return pd.Series(values, index=columns)

def no_answer(df):
    return df[~df['response'].isin(QUESTION_STRINGS)]

def print_question(example):
    gameState = pm.decode_gameState(example['scenario_json'])
    context_name = model_experiment.ORDER[example['question']]
    question = model_experiment.question_to_dialogue(gameState, context_name)
    print("SCENARIO:")
    print(example['text'])
    print("-----")
    output = f"NBS: {example['scenario_nbs']}; MEC: {example['scenario_mec']}"
    if 'scenario_mft' in example:
        output += f"; MFT: {example['scenario_mft']}"
    print(output)

def compare_llm_and_scenario(example):
    print_question(example)
    print("RESPONSE:")
    print(example['text'])

# todo: need functions which take the experimental conditions to compare and
# then also the model, or mturk as an argument. Want to be able to return these
# dataframes and then call other functions using those dfs as arguments (e.g. with names
# passed as strings as well)

# TODO: these could accept a wildcard argument...
# def get_conditions(temp=None,
#             api=None,
#             qualification=None,
#             show_charts=None,
#             chart_type=None
#             samples=None,
#             zero_shot=False):

def llm_dfs(scenario, desired_condition=None, desired_model=None, print_size=False, qualification=None, aggregate_responses=True):
    result = []
    for model in os.listdir(LLM_RESULTS_DIR):
        if model in ["beaker", ".DS_Store"] or (desired_model is not None and model != desired_model) or model not in MODEL_NAMES_SHORT:
            continue
        model_dir = os.path.join(LLM_RESULTS_DIR, model)
        model_run_dir = os.path.join(model_dir, scenario) # todo: should assert exists
        if not os.path.isdir(model_run_dir):
            continue
        for condition in os.listdir(model_run_dir):
            filename = os.path.join(model_run_dir, condition)
            if condition in ['.DS_Store'] or os.path.isdir(filename):
                continue
            if qualification is not None and f"qualification={qualification}" not in condition:
                continue
            if desired_condition is None or desired_condition in condition: # because the latter ends in .csv
                # TODO: do the exploding here? Probably, right?
                raw = pd.read_csv(filename, dtype=input_dtypes(num_scenarios))
                df = llm_preprocess_df(raw)
                df = add_fair(df)
                df = add_rawls(df)
                df = add_fehr(df, alpha=INEQUALITY_SUPPORT)
                cut = len(raw) - len(df)
                if print_size:
                    print(model)
                    print(condition)
                    print(f'{cut}/{cut/len(raw):.2f} total: {len(raw)}')
                df.name = MODEL_NAMES_SHORT[model]

                dist = df
                if aggregate_responses:
                    dist = get_response_dist(df, ['scenario_hash'])
                    dist.name = df.name

                if desired_model is not None:
                    dist.name = shorter_llm_condition(condition)
                result.append(dist)
    return result

def mturk_dfs(scenario, num_scenarios, desired_condition=None,
              delete_user_fail_check=False, filter_control=False,
             aggregate_responses=True, plot=True):
    run_dir = os.path.join(MTURK_RESULTS_DIR, scenario)
    scenario_file = os.path.join(SCENARIOS_DIR, scenario + ".csv")
    scenario_df = pd.read_csv(scenario_file)
    scenario_df = explode_df(scenario_df, num_scenarios)

    result = []
    failed_checks = {'responses' : {},
                     'workers' : {}}
    all_workers_failed = 0

    times = []
    for condition in os.listdir(run_dir):
        filename = os.path.join(run_dir, condition)

        if condition in [".DS_Store"] or "duplicate" in condition or os.path.isdir(filename):
            continue
        if desired_condition is None or desired_condition in condition: # because the latter ends in .csv
            df = pd.read_csv(filename, dtype=input_dtypes(num_scenarios), na_values=["{}"])
            times.append(df['Answer.tm'])
            num_workers = len(df)
            df = mturk_explode_df(df, num_scenarios)
            # Have to replace the 'scenario_hash' because for some reason it has been treated as a float and rounded in some places
            copy = df.copy()
            del copy['scenario_hash']
            json_only = scenario_df[['scenario_hash', 'scenario_json']].copy().drop_duplicates()
            df = pd.merge(copy, json_only, on=['scenario_json'], how='inner')
            
            mturk_df = (df[df['attention-answer'] == df['attention-response']])

            this_failed_checks = df[df['attention-answer'] != df['attention-response']]

            workers_failed = this_failed_checks['WorkerId'].unique()
            workers_passed = mturk_df[~mturk_df['WorkerId'].isin(workers_failed)]
                
            name = MTURK_CONDITIONS_SHORT[condition]
            failed_checks['responses'][name] = (len(df) - len(mturk_df)) / len(df)
            failed_checks['workers'][name] = (len(df) - len(workers_passed)) / len(df)
            print(f"{name}: {len(workers_failed)}/{len(workers_failed) / num_workers:.2f} failed checks out of {num_workers}")
            all_workers_failed += len(workers_failed)

            if delete_user_fail_check:
                mturk_df = workers_passed
            
            if filter_control:
                mturk_df = workers_pass_control(df)

            df = mturk_df
            df = add_fehr(df, alpha=INEQUALITY_SUPPORT)
            df = add_rawls(df)
            if aggregate_responses:
                df = get_response_dist(mturk_df, ['scenario_hash'])
            df.name = name
            result.append(df)

    if plot:
        pd.DataFrame(failed_checks).plot.bar(title="% Fail Attention Checks by Response or Worker")
    print(f"Total workers failed checks: {all_workers_failed}")

    report_time(times)

    return (result)

### plotting

def plot_random(axes, value=1/3, legend=False):
    xmin, xmax = axes.get_xlim()
    axes.hlines(y=value, xmin=xmin, xmax=xmax, colors='grey', linestyles='--', lw=2, label='Random')
    if legend:
        axes.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

def percent_correct_area_by_condition(group, name, selector=None, func=get_mec_percent):
    # TODO: waiting to fix this function until fixed the above bug with right qual data
    results = {'area' : pd.DataFrame(),
               'volume' : pd.DataFrame(),
               'both' : pd.DataFrame(),
               'show-charts=False' : pd.DataFrame()}

    for df in group:
        df_name = df.name
        if selector:
            df = selector(df)
        right_answer = df.apply(func, axis=1)
        for key in results.keys():
            if key in df_name:
                results[key] = pd.concat([results[key], right_answer], axis=1)

    for key in results.keys():
        results[key] = results[key].sum(axis=1)
        results[key].name = key
    

    pd.concat(results.values(), axis=1).plot.box(title=f"% correct qual. ans. -- {name}")

def _success(freq, n, normalize, prior=1/3, confidence=.95):
    if n == 0:
        return 0, (0, 0)
    p = freq / n

    test = scipy.stats.binomtest(k=freq, n=n, p=prior, alternative='two-sided')
    ci = test.proportion_ci(confidence)

    low = (freq - (ci.low * n))
    high = ((ci.high * n) - freq)
    if normalize:
        return (p, (low / n, high / n))
    else: 
        return (freq, (low, high))

def control_success(df, normalize=True, between=('scenario_nbs', 'scenario_mec')):
    theory_one = between[0]
    theory_two = between[1]
    frequency = len(df[(df['response'] == df[theory_one]) & (df['response'] == df[theory_two])])
    return _success(frequency, len(df), normalize)

def function_success(df, function, normalize=True):
    frequency = len(df[(df['response'] == df[function])])
    return _success(frequency, len(df), normalize)
    
def plot_conditions_distribution(dfs, selector, title, axis=True):
    decrease = 0
    if not axis:
        decrease = 1
    fig, ax = plt.subplots(ncols=len(dfs), sharey=True, figsize=(LLM_TEST_PLOT_WIDTH * len(dfs) - decrease, 4.8))
    fig.suptitle(title, fontsize=14)
    if len(dfs) > 1:
        first_ax = ax[0]
    else:
        first_ax = ax
    # fig.supxlabel('Aggregation Method')
    if axis:
        first_ax.set_ylabel("% agreement")

    dfs = sort_by_condition(dfs)

    for i in range(len(dfs)):
        df = dfs[i]
        mec = (selector(df)).apply(get_mec_percent, axis=1)
        mec.name = mec_short

        nbs = (selector(df)).apply(get_nbs_percent, axis=1)
        nbs.name = nbs_short
        condition = df.name

        if len(condition) > 10:
            condition = shorter_llm_condition(df.name)
        if len(dfs) > 1:
            this_ax = ax[i]
        else:
            this_ax = ax
        _, bplot = pd.concat([mec.to_frame(), nbs.to_frame()])\
          .plot.box(title=condition, ax=this_ax, widths=BOX_PLOT_WIDTH, ylim=PERCENT_PLT_YLIM,
                   patch_artist=True, return_type='both')
        for patch, color in zip(bplot['boxes'], list(AGGREGATION_TO_COLOR.values())):
            patch.set_facecolor(color)
        plot_random(this_ax)
        if not axis:
            this_ax.tick_params(axis="y", labelleft=False, left=False)

def plot_condition_binomial(df, selector, ax, axis=True, blank=False, between=('scenario_mec', 'scenario_nbs')):
    result = {}

    lower_errors = []
    upper_errors = []

    for function in between:
        datum, (l_error, u_error) = function_success(selector(df, between=between), normalize=True, function=function)
        result[FUNCTION_NAMES_TO_SHORT[function]] = datum
        lower_errors.append(l_error)
        upper_errors.append(u_error)

    condition = df.name

    if len(condition) > 10:
        condition = shorter_llm_condition(df.name)

    # # combine and reshape
    errors = [lower_errors, upper_errors]

    data_df = pd.DataFrame(list(result.items()),columns = ['Label','Value'])

    plot_title = condition

    colors = [AGGREGATION_TO_COLOR[function] for function in between]
    
    data_df.plot.bar(x='Label', y='Value',
                     title=plot_title,
                     ylim=(0, 1.01), 
                     ylabel=f"% Agreement" if axis else "",
                     xlabel="",
                     ax=ax,
                     yerr=errors,
                     color=colors)
    if blank:
        clear_plot_data(ax)
    plot_random(ax)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)
    ax.get_legend().remove()
    
    ax.tick_params(axis="y", labelleft=False, left=False)

def plot_conditions_binomial(dfs, selector, title, axis=True, blank=False, axes=None, **kwargs):
    decrease = 0
    fig = None
    if not axis:
        decrease = .5
    if axes is None:
        fig, axes = plt.subplots(ncols=len(dfs), sharey=True, figsize=(1 + LLM_TEST_PLOT_WIDTH * len(dfs) - decrease, 4.8))
        fig.suptitle(title, fontsize=14)
    # fig.supxlabel('Aggregation Method')
    first_ax = axes[0] if len(dfs) > 1 else axes
    if axis:
        first_ax.set_ylabel("% agreement")
    
    dfs = sort_by_condition(dfs)
    
    for i in range(len(dfs)):
        df = dfs[i]
        this_ax = axes[i] if len(dfs) > 1 else axes
        plot_condition_binomial(df, selector, this_ax, axis, blank, **kwargs)
    
    return fig

def plot_control_binomial(dfs, title, ax=None, blank=False, **kwargs):
    result = {}
    errors = {}
    for df in dfs:
        name = df.name
        df = not_null_ties(control(df, **kwargs))
        success, (low, high) = control_success(df, **kwargs)
        result[name] = success
        errors[name] = np.array([low]), np.array([high])
    
    yerrs = [[val[0][0] for val in errors.values()], [val[1][0] for val in errors.values()]] 
    
    data_df = pd.DataFrame(list(result.items()),columns = ['Label','Value'])

    figsize = None if ax else (1 + ((3 / 4) * len(dfs)),4.8)

    color = list(CONDITION_TO_COLOR.values())
    if len(dfs) < 2:
        color = color[3:] # None condition


    ax = data_df.plot.bar(x='Label', y='Value',
                     title=title,
                     figsize=figsize,
                     ylim=(0, 1.01), 
                     ylabel=f"% Agreement",
                     xlabel="",
                     ax=ax,
                     yerr=yerrs,
                     color=color)

    if blank:
        clear_plot_data(ax)
    # if len(dfs) == 1:
    #     ax.set_xticks([])
    
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)
    plot_random(ax)
    ax.get_legend().remove()
    return ax

def space_last_fig_add_titles(fig, axes):
        
    extra_space = 0.08
    
    # Adjust the position of the last subplot to maintain its width
    pos_last = axes[-1].get_position()
    axes[-1].set_position([pos_last.x0 + extra_space, pos_last.y0, pos_last.width, pos_last.height])
    
    
    top_row_suptitle = fig.text(0.43, 1, 'default', ha='center', fontsize=14)
    
    top_row_suptitle = fig.text(.915, 1, 'random', ha='center', fontsize=14)

def confidence_interval_95(data):
    bootstrap = scipy.stats.bootstrap((data,), np.mean)
    lower, upper = data.mean() - bootstrap.confidence_interval.low, bootstrap.confidence_interval.high - data.mean()
    return lower, upper
    
def control_agreement_stats(group):
    aggregared = group.apply(lambda x: x['response'] == x['scenario_nbs'], axis=1)
    lower, upper = confidence_interval_95(aggregated)
    return pd.DataFrame({'mean' : aggregated.mean(), 'error' : {'lower' : lower, 'upper' : upper} }, index=[group.iloc[0]['scenario_hash']])

PERCENT_AGREE = f"Percent Agree with {nbs_long}, {mec_long}"

def plot_mean_control_agreement_scenario(dfs):
    data = {}
    errors = {}
    for df in dfs:
        condition = df.name
        df['scenario_hash'] = df['scenario_hash'].astype('str')
        df = not_null_ties(control(df))
    
        grouped = df.groupby(['scenario_hash'])
    
        nbs_agree = lambda x: x['response'] == x['scenario_nbs']
        mean = grouped.apply(lambda g: g.apply(nbs_agree, axis=1).mean())

        error = grouped.apply(lambda g: np.array(confidence_interval_95(g.apply(nbs_agree, axis=1))).mean())

        data[condition] = mean

        errors[condition] = error
        # TODO: currently just averaging the upper and lower bounds b/c it is such a pain to show them separate
        # errors[condition] = np.array(error.apply(lambda x: [[x[0]], [x[1]]]).to_list())

    err = pd.DataFrame(errors)
    df =  pd.DataFrame(data)

    ax = df.plot.bar(yerr=err, color=list(CONDITION_TO_COLOR.values()),
                                figsize=(12, 4.8), ylabel=PERCENT_AGREE, xlabel="Non-disagreement (Control) Scenarios")
    plot_random(ax)

def plot_mean_control_agreement(dfs):
    data = {}
    error = []
    for df in dfs:
        condition = df.name
        df = not_null_ties(control(df))
        nbs_agree = lambda x: x['response'] == x['scenario_nbs']
        mean = df.apply(nbs_agree, axis=1).mean()
        lower, upper = confidence_interval_95(df.apply(nbs_agree, axis=1))
        data[condition] = {PERCENT_AGREE: mean}
        error.append([[lower], [upper]])
    
    data_df = pd.DataFrame(data)
    errors = np.array(error)

    ax = data_df.plot.bar(yerr=errors, color=list(CONDITION_TO_COLOR.values()), xlabel="Condition", ylabel=PERCENT_AGREE,
                    figsize=(2, 4.8))
    plot_random(ax)
    plt.xticks(rotation=0)
    # NB: if these truly need to be separate columns on the plot have to transpose the matrix and then figure
    # something out for color, ie `data_df.transpose().plot.bar(c=['black', 'red', 'blue'])`

def clear_plot_data(ax):
    # Remove all lines
    for line in ax.lines:
        line.remove()
    # Remove all collections (e.g., scatter plots)
    for collection in ax.collections:
        collection.remove()
    for patch in ax.patches:
        patch.remove()
    # Redraw the plot
    ax.figure.canvas.draw()

def save_fig(fig, name):
    plt.savefig(f"figures/{name}.pdf", bbox_inches='tight')
    plt.savefig(f"figures/{name}.svg", bbox_inches='tight', transparent=True)
    plt.tight_layout()
    plt.show()

####

def get_all_llm_dfs(scenario):
    c1 = "temp=1_api=chat_qualification=False_show-charts=False_chart-type=area_samples=10_zero-shot=False.csv"
    c2 = "temp=1_api=completion_qualification=False_show-charts=False_chart-type=area_samples=10_zero-shot=False.csv"
    
    llm_dfs_no_chart = llm_dfs(scenario, desired_condition=c1)
    llm_dfs_no_chart += llm_dfs(scenario, desired_condition=c2)
    
    llm_dfs_no_chart_na = llm_dfs(scenario, desired_condition=c1, aggregate_responses=False)
    llm_dfs_no_chart_na += llm_dfs(scenario, desired_condition=c2, aggregate_responses=False)
    
    c1 = "temp=1_api=chat_qualification=False_show-charts=True_chart-type=area_samples=10_zero-shot=False.csv"
    c2 = "temp=1_api=completion_qualification=False_show-charts=True_chart-type=area_samples=10_zero-shot=False.csv"
    
    llm_dfs_area_chart = llm_dfs(scenario, desired_condition=c1)
    llm_dfs_area_chart += llm_dfs(scenario, desired_condition=c2)
    
    llm_dfs_area_chart_na = llm_dfs(scenario, desired_condition=c1, aggregate_responses=False)
    llm_dfs_area_chart_na += llm_dfs(scenario, desired_condition=c2, aggregate_responses=False)
    
    c1 = "temp=1_api=chat_qualification=False_show-charts=True_chart-type=volume_samples=10_zero-shot=False.csv"
    c2 = "temp=1_api=completion_qualification=False_show-charts=True_chart-type=volume_samples=10_zero-shot=False.csv"
    
    llm_dfs_volume_chart = llm_dfs(scenario, desired_condition=c1)
    llm_dfs_volume_chart += llm_dfs(scenario, desired_condition=c2)
    
    llm_dfs_volume_chart_na = llm_dfs(scenario, desired_condition=c1, aggregate_responses=False)
    llm_dfs_volume_chart_na += llm_dfs(scenario, desired_condition=c2, aggregate_responses=False)
    
    c1 = "temp=1_api=chat_qualification=False_show-charts=True_chart-type=both_samples=10_zero-shot=False.csv"
    c2 = "temp=1_api=completion_qualification=False_show-charts=True_chart-type=both_samples=10_zero-shot=False.csv"
    
    llm_dfs_both_chart = llm_dfs(scenario, desired_condition=c1)
    llm_dfs_both_chart += llm_dfs(scenario, desired_condition=c2)
    
    
    llm_dfs_both_chart_na = llm_dfs(scenario, desired_condition=c1, aggregate_responses=False)
    llm_dfs_both_chart_na += llm_dfs(scenario, desired_condition=c2, aggregate_responses=False)
    ##

    models_to_dfs = {}
    models_to_dfs_no_agg = {}
    
    for model in MODELS:
        models_to_dfs[model] = []
        models_to_dfs_no_agg[model] = []
    
    for df in llm_dfs_area_chart:
        model = df.name
        new_df = df.copy()
        new_df.name = 'Area'
        models_to_dfs[model].append(new_df)
    
    for df in llm_dfs_volume_chart:
        model = df.name
        new_df = df.copy()
        new_df.name = 'Volume'
        models_to_dfs[model].append(new_df)
    
    for df in llm_dfs_both_chart:
        model = df.name
        new_df = df.copy()
        new_df.name = 'Both'
        models_to_dfs[model].append(new_df)
    
    for df in llm_dfs_no_chart:
        model = df.name
        new_df = df.copy()
        new_df.name = 'None'
        models_to_dfs[model].append(new_df)
    
    ###
    
    for df in llm_dfs_area_chart_na:
        model = df.name
        new_df = df.copy()
        new_df.name = 'Area'
        models_to_dfs_no_agg[model].append(new_df)
    
    for df in llm_dfs_volume_chart_na:
        model = df.name
        new_df = df.copy()
        new_df.name = 'Volume'
        models_to_dfs_no_agg[model].append(new_df)
    
    for df in llm_dfs_both_chart_na:
        model = df.name
        new_df = df.copy()
        new_df.name = 'Both'
        models_to_dfs_no_agg[model].append(new_df)
    
    for df in llm_dfs_no_chart_na:
        model = df.name
        new_df = df.copy()
        new_df.name = 'None'
        models_to_dfs_no_agg[model].append(new_df)

    return (models_to_dfs, models_to_dfs_no_agg)

# load in all of the quals, 
def qual_stats(df):
    with open("mturk/qualification_answers.json", 'r') as qualJson:
        qualification_answers = json.load(qualJson) 
        qualification_answers = {f'Answer.{k}' : v for k, v in qualification_answers.items()}

    perfect_scores = 0
    for _, row in df.iterrows():
        score = 0
        for question, answer in qualification_answers.items():
            score += row[question] == answer
        if score == 13:
            perfect_scores += 1
    return (perfect_scores, len(df))

def report_time(times):
    all_times = pd.concat(times, ignore_index=True)
    all_times = all_times.div(60)
    print(f'total time in minutes: mean {all_times.mean():.1f} and std {all_times.std():.1f} and wage ${ (60 / all_times.mean()) * 3:.1f}')


def row_to_beliefs(row):
    prb_str = 'probability: '
    prob_columns = list(filter(lambda x: x.startswith(prb_str), list(row.keys())))
    beliefs = sorted([0 if np.isnan(row[col]) else row[col] for col in prob_columns], reverse=True)
    return beliefs

def percent_by_position(row, position):
    # TODO: could move this function out later for row to dist
    return row_to_beliefs(row)[position]

def get_answer_percent(row):
    return row[f"probability: {row['response']}"]

###

def model_agree_stats_by_group(group, name, order=None, selector=None, percentage=False):

    result = []
    for df in group:
        df_name = df.name
        if selector:
            df = selector(df)
        agree = model_agree_stats(df, percentage=percentage)
        agree.name = df_name
        result += [agree.to_frame()]

    df = pd.concat(result, axis=1)
    if order:
        df.reindex(order, axis=1)

    prefix="#"
    if percentage:
        prefix='%'
    df.plot.bar(title=f"{prefix} Support -- {name}")

def model_agree_stats_both_conditions(group, name, percentage=True):
    
    selectors = {"control" : control, "test" : test, }
    
    fig = None
    ax = None
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(6, 4.8), width_ratios=(1, 2))
    
    prefix = '#'
    ylim=None

    if percentage:
        prefix= '%'
        ylim=(0,1)
    
    fig.suptitle(f'{prefix} Support -- {name}', fontsize=16)
    
    result_dfs = []
    
    for i in range(len(selectors)):
        selector_name = list(selectors.keys())[i]
        selector = selectors[selector_name]

        results = {'Area' : [],
               'Volume' : [],                   
               'Both' : [],
               'None' : []}
        for df in group:
            df_name = df.name
            df = selector(df)
            if 'qualification=True' in df_name:
                continue
            if 'show-charts=False' in df_name:
                df_name = 'none'
            agree = model_agree_stats(df, percentage)
            
            if selector_name == 'control':
                agree[f'{mec_short}/{nbs_short}'] = agree[mec_short]
                del agree[mec_short]
                del agree[nbs_short]
            for key in results.keys():
                if key in df_name:
                    results[key].append(agree)
        
        dataframes = {}
        
        # TODO: this should no longer be necessary
        for key in results.keys():
            if len(results[key]) < 1:
                continue
            dataframes[key] = pd.concat(results[key], axis=1).div(len(results[key])).sum(axis=1)
            dataframes[key].name = key
        if len(dataframes) > 0:
            result = pd.concat(dataframes.values(), axis=1)
    
            result.plot.bar(title=selector_name, ylim=ylim, ax=ax[i], legend=None)
            ax[i].set_xticks(ax[i].get_xticks(), ax[i].get_xticklabels(), rotation=0)
                                       
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
#     plt.xticks(rotation=0)

###

## TODO: verify table is formatted correctly, do the same for the llms

def significance_to_stars(significance):
    if significance < .001:
        return "***"
    elif significance < .01:
        return "**"
    elif significance < .05:
        return "*"
    return ""
        
def stat_tests_and_means(dfs, function=test):
    table = {}
    for df in dfs:
        table[df.name] = {}
        mec = function(df).apply(get_mec_percent, axis=1)
        mec_count = len(function(df)) * mec.mean()
        nbs = function(df).apply(get_nbs_percent, axis=1)
        nbs_count = len(function(df)) * nbs.mean()
        ttest = scipy.stats.ttest_ind(mec, nbs, nan_policy='omit')

        nbs_random = scipy.stats.ttest_1samp(nbs, 1/3, nan_policy='omit')
        mec_random = scipy.stats.ttest_1samp(mec, 1/3, nan_policy='omit')

        table[df.name][nbs_long] = f"{nbs_count: .1f} / {nbs.mean(): .2f}{significance_to_stars(nbs_random.pvalue)}"
        table[df.name][mec_long] = f"{mec_count: .1f} / {mec.mean(): .2f}{significance_to_stars(mec_random.pvalue)}"
        table[df.name][f'{mec_short}-{nbs_short}'] = significance_to_stars(ttest.pvalue)

    print(pd.DataFrame(table).to_latex())

def stat_tests_and_means_counts_test(dfs):
    table = {}
    for df in dfs:
        name = df.name
        table[name] = {}
        df = test(df)
        n = len(df)
        prior = 1/3
        util_yes = len(df[(df['response'] == df['scenario_mec'])])

        nash_yes = len(df[(df['response'] == df['scenario_nbs'])])

        util_p = scipy.stats.binomtest(k=util_yes, n=n, p=prior, alternative='two-sided').pvalue
        
        nash_p = scipy.stats.binomtest(k=nash_yes, n=n, p=prior, alternative='two-sided').pvalue

        table[name][nbs_long] = f"{nash_yes} / {n}{significance_to_stars(nash_p)}"
        table[name][mec_long] = f"{util_yes} / {n}{significance_to_stars(util_p)}"

    print(pd.DataFrame(table).to_latex())

def stat_tests_and_means_counts_control(dfs):
    table = {}
    for df in dfs:
        name = df.name
        table[name] = {}
        df = not_null_ties(control(df))
        n = len(df)
        prior = 1/3
        observations = len(df[(df['response'] == df['scenario_mec']) & (df['response'] == df['scenario_nbs'])])

        p_value = scipy.stats.binomtest(k=observations, n=n, p=prior, alternative='two-sided').pvalue

        table[name][f'{mec_short}&{nbs_short}'] = f"{observations} / {n}{significance_to_stars(p_value)}"

    print(pd.DataFrame(table).to_latex())