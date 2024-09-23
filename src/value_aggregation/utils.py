import base64
import itertools
import functools
import collections
import logging
import math
import numpy as np
import string
import json
import random
import sys

logger = logging.getLogger(__name__)


from .GameState import VoteGameState
from .Game import run_mec, run_nash_bargain, run_mft, run_equality_efficiency

# TODO: move more functions here

def add_dict_keys_to_lists(beliefs, outcomes, agents, actions):
    """
    Assumes beliefs in the shape of (num_agents,)
    Assumes outcomes in the shape of (num_actions, num_agents)
    Represents agents as supplied by agents and actions
    
    Takes a list of beliefs and a nested list of outcomes.
    Returns a tuple of a new dict with letters as the keys to beliefs
    and numbers as the keys to the outcomes.
    """
    num_agents = len(beliefs)
    num_actions = len(outcomes)
    
    beliefs_dict = dict(zip(agents, beliefs))
    outcomes_dict = {}
        
    for key, action_outcomes in zip(actions, outcomes):
        outcomes_dict[key] = dict(zip(agents, action_outcomes))
    
    return (beliefs_dict, outcomes_dict)

def normalize(group):
    total = sum(group)
    result = frozenset(map(lambda y: y / total, group))
    return result

def denormalize(group, num_agents, belief_norm):
    # For the case when the set(.5, .5) => (.5), e.g.
    if len(group) < num_agents:
        group = [1/num_agents] * num_agents
    # Rounding because the outputs need to be human readable
    return [round(belief * belief_norm, 0) for belief in group]

def not_all_equal(group):
    return len(set(group)) != 1

def make_belief_range(belief_range, belief_range_steps, num_agents, belief_norm):
    agent_interval = np.linspace(*belief_range, belief_range_steps)
    beliefs = itertools.product(agent_interval, repeat=num_agents)

    # Need to normalize the beliefs to get rid of duplicates
    # But we want to unnormalize for legibility

    normalized = set(map(normalize, beliefs))

    denormalize_partial = functools.partial(denormalize, num_agents=num_agents, belief_norm=belief_norm)

    denormalized = map(denormalize_partial, normalized)

    return denormalized

def generate_games(agents, actions, belief_range_steps=2,
                   action_range_steps=2, action_range=(1,101),
                   belief_range=(.01, .99), belief_norm=100,
                   action_range_func=np.linspace,
                   gameState=VoteGameState, max_actions=100):
    """
    Generates games with `num_agents` agents and `num_actions` actions
    where beliefs for agents range from belief_range * belief_norm 
    [1, 99] by default with `belief_range_steps` 
    number of beliefs sampled uniformly and where outcomes for actions range
    from `[action_range[0], action_range[1]]` with `action_range_steps` number
    of outcomes sampled uniformly.

    Returns a list of `gameState` objects for each of the corresponding games.

    action_range_func could be np.linspace or np.logspace

    Set `action_range_steps` to `abs(action_range[0] - action_range[1]) + 1`
    to sample the whole numbers.
    """
    num_agents = len(agents)
    num_actions = len(actions)

    beliefs = make_belief_range(belief_range, belief_range_steps, num_agents, belief_norm)

    action_interval = action_range_func(*action_range, action_range_steps)
    agents_by_actions = itertools.product(action_interval, repeat=num_agents)
    if max_actions is not None:
        agents_by_actions = list(agents_by_actions)
        agents_by_actions = random.sample(agents_by_actions, min(max_actions, len(agents_by_actions)))
    outcomes = itertools.product(agents_by_actions, repeat=num_actions)


    # Filter out when all the outcomes are the same
    new_outcomes = list(filter(not_all_equal, outcomes))

    # But add back at least one example of this
    equal_outcome = ((action_interval[-1],) * num_agents,) * num_actions
    new_outcomes.append(equal_outcome)
    
    games = list(itertools.product(beliefs, new_outcomes))
    
    game_states = [gameState(*add_dict_keys_to_lists(*game, agents, actions)) for game in games]
    
    return game_states

def split_by(source, length):
    samples = []
    for i in range(len(source)):
        sample = []
        for j in range(length):
            index = (i + j) % len(source)
            sample.append(source[index])
        samples.append(sample)
    return samples

def generate_game(agent_interval, agents, action_interval, actions, gameState):
    num_agents = len(agents)
    num_actions = len(actions)
    beliefs = tuple(random.choices(agent_interval, k=num_agents))
    # for each agent...
    outcomes = ()
    for _ in range(num_agents):
        outcomes += (tuple(random.choices(action_interval, k=num_actions)),)
    game = (beliefs, outcomes)
    return gameState(*add_dict_keys_to_lists(*game, agents, actions)) 

def generate_hits_greedy(agents, actions, belief_range_steps=2,
                         action_range_steps=2, action_range=(1,101),
                         belief_range=(.01, .99), belief_norm=100,
                         action_range_func=np.linspace,
                         gameState=VoteGameState,
                         scenarios_per_hit=3,
                         sample_size=500,
                         aggregation_functions=[run_mec, run_nash_bargain],
                         allow_ties=True,
                         disagrees_only=False):

    agent_interval = list(map(math.floor, np.linspace(*belief_range, belief_range_steps) * belief_norm))
    action_interval = action_range_func(*action_range, action_range_steps)

    set_sample_size = math.floor(sample_size / 2)
    games_generated = 0
    disagreements = set()
    non_disagreements = set()
    while len(disagreements) < set_sample_size or len(non_disagreements) < set_sample_size:
        game = generate_game(agent_interval, agents, action_interval, actions, gameState)
        games_generated += 1
        if games_generated > 20000:
            logging.error("Generated 20000 games without exiting loop. Exiting")
            sys.exit(1)

        if not allow_ties and not is_not_tie(game, aggregation_functions):
            continue # This game is a tie and we do not want ties
        disagreement = disagrees_func(game, aggregation_functions)
        if disagreement and len(disagreements) < set_sample_size:
            disagreements.add(game)
        elif not disagreement and len(non_disagreements) < set_sample_size:
            non_disagreements.add(game)

    non_disagreements_labeled = map(encode_hit, non_disagreements)
    disagreements_labeled = map(encode_hit, disagreements)

    all_scenarios = list(disagreements_labeled)
    if not disagrees_only:
        all_scenarios += list(non_disagreements_labeled)
    random.shuffle(all_scenarios)

    hit_list = split_by(all_scenarios, scenarios_per_hit)

    random.shuffle(hit_list)

    return hit_list

# quantities is redundant and expensive.
def to_counts(gameState, aggregation_functions):
    results = []
    for function in aggregation_functions:
        results.append(function(gameState=gameState, none_if_tie=True))

    counts = collections.Counter(results)
    return counts

def is_not_tie(gameState, aggregation_functions):
    counts = to_counts(gameState, aggregation_functions)
    return counts[None] == 0 and (len(counts) == 1 or len(counts) == len(aggregation_functions))

def disagrees_func(gameState, aggregation_functions):
    counts = to_counts(gameState, aggregation_functions)
    return counts[None] == 0 and len(counts) == len(aggregation_functions)

def encode_hit(gameState):
    # False on Ascii so as not to have to decode Unicode
    sce_bytes = encode_gameState(gameState)
    result = {'hash' : hash(gameState),
            'json' : sce_bytes,
            'mec' : run_mec(gameState, none_if_tie=True),
            'nbs' : run_nash_bargain(gameState, none_if_tie=True),
            'mft' : run_mft(gameState, none_if_tie=True), 
            'fehr' : run_equality_efficiency(gameState, none_if_tie=True, alpha=.3)}
    assert(result['mec'] == run_mec(gameState, none_if_tie=True))
    assert(result['nbs'] == run_nash_bargain(gameState, none_if_tie=True))
    return result

def generate_hits(games, scenarios_per_hit=3, sample_size=500,
                  aggregation_functions=[run_mec, run_nash_bargain],
                  allow_ties=True, disagrees_only=False):

    games = set(games)

    logger.info(f'number of scenarios: {len(games)}')

    # TODO: Filtering in this way and then later calculating these

    is_not_tie_partial = functools.partial(is_not_tie, aggregation_functions=aggregation_functions)
    disagrees_func_partial = functools.partial(disagrees_func, aggregation_functions=aggregation_functions)    

    if not allow_ties:
        games = set(filter(is_not_tie_partial, games))

    disagreements = set(filter(disagrees_func_partial, games))

    logger.info(f'number of disagreements: {len(disagreements)}')

    logger.info(f'percent disagreements: {len(disagreements) / len(games):.4f}')

    no_disagreements = games - disagreements

    total_sample_size = min(sample_size, len(disagreements) * 2)
    set_sample_size = math.floor(total_sample_size / 2)

    disagreements_sample = random.sample(list(disagreements), set_sample_size)

    # Sample the same number from remaining
    no_disagreements_sample = random.sample(list(no_disagreements), set_sample_size)

    no_disagreements_labeled = map(encode_hit, no_disagreements_sample)
    disagreements_labeled = map(encode_hit, disagreements_sample)

    all_scenarios = list(disagreements_labeled)
    if not disagrees_only:
        all_scenarios += list(no_disagreements_labeled)
    random.shuffle(all_scenarios)

    hit_list = split_by(all_scenarios, scenarios_per_hit)

    random.shuffle(hit_list)

    return hit_list

def encode_gameState(gameState):
    as_json = json.dumps(gameState.toArray(), ensure_ascii=False)
    return base64.b64encode(bytes(as_json, 'utf-8')).decode('utf8')

def decode_gameState(encoded):
    gameState_json = base64.b64decode(bytes(encoded, 'utf-8')).decode('utf-8')
    return VoteGameState.fromArray(json.loads(gameState_json))
