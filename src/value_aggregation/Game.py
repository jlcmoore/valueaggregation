from collections import Counter
import logging
import numpy as np
import pygambit as gb
import math
import itertools
import functools

from .GameState import (VoteGameState, GameAction, ProportionalChancesGameState,
                        normalizeOutcomes, agentRangeNormalizeOutcomes)

logger = logging.getLogger(__name__)

######## Minimax game selection

def expectiMax(gameState, depth, agentIndex, chanceNode=True):
    """
    This is the maximization node. It calculates which action the particular agent should 
    take, the action which has the maximum utility.
    """
    if depth == 0:
        return (gameState.scoreGame(), gameState.getActionsTaken())

    # Not checking against a negative infinity vector because there might be such values allowed
    bestUtilityVector = None
    bestVotes = None

    for action in gameState.getLegalActions(agentIndex):
        nextState = gameState.generateSuccessor(agentIndex, action)
        nextAgentIndex = agentIndex + 1

        if chanceNode and nextAgentIndex >= gameState.numAgents():
            nextUtilityVector, nextVotes = expectiChance(nextState, depth, nextAgentIndex)
        else:
            tmp_depth = depth
            if not chanceNode and nextAgentIndex >= gameState.numAgents():
                tmp_depth -= 1
            nextUtilityVector, nextVotes = expectiMax(nextState, tmp_depth, nextAgentIndex, chanceNode)
        
        if bestUtilityVector is None or nextUtilityVector[agentIndex] > bestUtilityVector[agentIndex]:
            bestUtilityVector = nextUtilityVector
            bestVotes = nextVotes

    tabs = (gameState.numAgents() - agentIndex) * '\t'
    logger.debug(f"{tabs}{gameState.getAgent(agentIndex)}: {bestUtilityVector}, {bestVotes}")
    
    return (bestUtilityVector,  bestVotes)

def expectiChance(gameState, depth, agentIndex):
    """
    This is the "proportional chances" voting node. It calculates what each agent has voted
    for so far and weights the actions by the total votes received relative to the credences.
    """
    if agentIndex < gameState.numAgents():
        raise ValueError("Should only be called after all players.")
    if depth == 0:
        return (gameState.scoreGame(), gameState.getActionsTaken())

    expectedValueVector = gameState.payoffFromVotes(gameState.getVotesTaken())

    logger.debug(f"chance: {expectedValueVector}")
    return (expectedValueVector, gameState.getActionsTaken())

def run_expectimax(gameState):
    """
    Tallies the votes of the game passed in and returns the action with the most votes.
    Returns None if no action has more than 50% of the votes.
    """
    # TODO: see example below about issues with ties for the max
    useChanceNode = type(gameState) is not VoteGameState
    utilities, actions = expectiMax(gameState, 1, 0, useChanceNode)
    votes_to_probabilities = gameState._actions_to_probabilities(actions)
    
    max_action = max(votes_to_probabilities, key=votes_to_probabilities.get)
    max_action_percent = votes_to_probabilities[max_action]
    
    logger.debug(f"Max action: {max_action} with {max_action_percent:.2%}")
    logger.debug(f"Tally: {votes_to_probabilities}")
    logger.debug(f"Players: {gameState.getAgents()}, actions: {actions}, utilities: {utilities}")
    
    if max_action_percent <= .5:
        return None
    return max_action

######## End minimax game selection

######## Equilibrium selection

def run_equilibrium_selection(gameState):
    """
    Computes the Nash equilibria of the game and returns just one of them.
    """
    payoffs = to_agent_payoffs(gameState)
    profiles = compute_equilibria(payoffs)
    equilibrium = choose_equilibrium(profiles)

    logger.debug(f"profile: {profile}")
    return equilibrium

def choose_action(gameState, equilibrium):
    # TODO: this whole approach is busted
    # should really just do this as a pre-processor to other approaches
    # so first look only at the equilibria, then return a new gameState
    # to consider with only those options available
    for i in gameState.numAgents():
        for j in gameState.numVotes():
            play_prob = equilibrium[(i * gameState.numAgents()) + j]

def to_agent_payoffs(gameState):
    """
    Converts the game state into a format readable by pygambit.
    Returns that format as a list of arrays of arrays.
    """
    gb_data_type = gb.Rational # to allow for fractional values
    # Get the cartesian product; all of the possible vote combinations
    all_votes = itertools.product(gameState.getVotes(), repeat=gameState.numAgents())
    # Determine the shape of the resulting strategic form matricies
    shape = tuple(itertools.repeat(gameState.numVotes(), gameState.numAgents()))
    # Compute the payoffs for each resulting combination of votes
    payoffs = map(gameState.payoffFromVotes, all_votes)
    # Convert the data type
    payoffs = map(lambda x: np.array(list(map(lambda y: gb_data_type(y), x))), payoffs)
    # Change the shape of the payoffs to fit the requirement
    payoffs_as_lists = map(lambda x: x.tolist(), payoffs)
    payoffs_per_agent = zip(*(payoffs_as_lists))
    payoffs_per_agent_arrays = list(map(lambda x: np.array(x, dtype=gb_data_type).reshape(shape), payoffs_per_agent))
    return payoffs_per_agent_arrays

def compute_equilibria(array_rep, mixed=True):
    """
    Given the correct representation for pygamit, returns the profiles
    according to the relevant equilibria solver.
    """
    num_players = len(array_rep)
    game = gb.Game.from_arrays(*array_rep)
    if mixed:
        if num_players == 2:
            solver = gb.nash.ExternalEnumMixedSolver()
        else:
            solver = gb.nash.ExternalSimpdivSolver()
    else: 
        solver = gb.nash.ExternalEnumPureSolver()
    profile = solver.solve(game)
    return profile

def choose_equilibrium(equilibria):
    """
    Returns the equilibrium with the highest total payoff over all players that
    has the highest credence-a compromise
    Often just returns the first profile.
    """

    # TODO: Require the chosen profile to be a compromise??
    # TODO: how to handle cases with no profiles? ... Mixed strategies, likely
    # TODO: just return the most payoff dominant profile? not sure if this is relevant
    choice = None
    max_utility = None
    for profile in equilibria:
        profile_sum = sum(profile.payoff())
        if choice is None or profile_sum > max_utility:
            choice = profile
            max_utility = profile_sum
    return choice

######## End equilibrium selection

def max_of_dict(values, none_if_tie=False):
    all_values = list(values.values())
    max_value = max(all_values)
    is_tie = all_values.count(max_value) > 1
    if none_if_tie and is_tie:
        return None
    arg_max = max(values, key=values.get)
    return arg_max

def run_nash_bargain(gameState, none_if_tie=False):
    """
    Returns the suggested vote as according to the asymmetric Nash Bargaining solution:

    $$\\argmax_{a \\in A} \\prod_{u_i \\in U} (u_i(a) − d_i )^{c_i}$$

    $$\\argmax_{a \\in A} \\sum_{u_i \\in U} c_i \\log_{10} (u_i(a) − d_i)$$

    Consult \\cite[eqn. 2,3]{greaves_bargaining_2022}. This is not a mixed strategy.
    """
    if type(gameState) is not VoteGameState:
        raise TypeError("Only accepts VoteGameStates.")

    gameState = agentRangeNormalizeOutcomes(gameState)

    votes_to_products = {}
    for vote in gameState.vote_agent_outcomes:
        votes_to_products[vote] = 1 # because doing *=
        for agent in gameState.getAgents():
            utility = gameState.vote_agent_outcomes[vote][agent]
            disagree_point = gameState.getDisagreementUtility(agent)
            credence = gameState.getNormalizedCredence(agent)
            votes_to_products[vote] *=  (utility - disagree_point) ** credence
            # Equivalent to:
            # votes_to_products[vote] +=  credence * math.log10(utility - disagree_point)

    logger.debug(f"Tally: {votes_to_products}")

    return max_of_dict(votes_to_products, none_if_tie)

def randomDictatorPolicy(gameState):
    """
    Takes a game, returns a utility vector according to:
    "We might set d_X = RD_X, the random dictator point, where, for each i, the act that
    is highest-ranked by T_i is selected with probability p_i."
    \\cite[pg. 13]{greaves_bargaining_2022}
    """
    votes = []
    for agent in gameState.getAgents():
        max_vote = None
        max_utility = None
        for vote in gameState.getVotes():
            utility = gameState.vote_agent_outcomes[vote][agent]
            if max_vote is None or utility > max_utility:
                max_vote = vote
                max_utility = utility

        votes.append(max_vote)

    return ProportionalChancesGameState.payoffFromVotes(gameState, votes)

def run_mft(gameState, none_if_tie=False):
    """
    Returns the suggested vote as according to My Favorite Theory---the suggested
    vote of the theory with the highest credence.
    """
    max_agent = max_of_dict(gameState.agents_to_credences, none_if_tie)
    if max_agent is not None:
        return gameState.getMaxVote(max_agent)
    else:
        return None

def run_mfo(gameState):
    """
    Returns the suggested vote as according to My Favorite Option---the suggested
    vote with the sum of the credences of those theories that find it permissible
    (has outcomes greater than zero)
    """
    votes_to_credences = {}
    for vote in gameState.vote_agent_outcomes:
        votes_to_credences[vote] = 0
        for agent in gameState.vote_agent_outcomes[vote]:
            outcome = gameState.vote_agent_outcomes[vote][agent]
            if outcome > 0:
                votes_to_credences[vote] += gameState.getNormalizedCredence(agent)
    
    logger.debug(f"Tally: {votes_to_credences}")
    return max(votes_to_credences, key=votes_to_credences.get)

def run_mec(gameState, none_if_tie=False):
    """
    Returns the suggested vote as according to Maximum Expected Choice-worthiness
    ---the vote with the sum of the outcomes proportional to the utilities of each 
    theory.
    """
    votes_to_utilities = {}
    for vote in gameState.vote_agent_outcomes:
        votes_to_utilities[vote] = 0
        for agent in gameState.vote_agent_outcomes[vote]:
            votes_to_utilities[vote] += (gameState.vote_agent_outcomes[vote][agent] 
                                            * gameState.getNormalizedCredence(agent))
    
    logger.debug(f"Tally: {votes_to_utilities}")

    return max_of_dict(votes_to_utilities, none_if_tie)

def run_leximin(gameState, none_if_tie=False):
    '''
    Returns the suggested vote as according to the egalitarian rule: maximizes the minimum utility 
    for the group, breaks ties with lexical minimum.
    '''
    if not gameState.equalCredences():
        raise ValueError("Have not implemented asymmetric version.")

    proposals = []
    vote_to_outcomes = gameState.vote_to_outcomes()

    outcomes = list(vote_to_outcomes.values())

    leximin_cmpt = make_comparator(leximin_less_than)
    leximin_cmpt_key = functools.cmp_to_key(leximin_cmpt)

    sorted_outcomes = sorted(outcomes, key=leximin_cmpt_key, reverse=True)

    max_outcome = sorted_outcomes[0]

    if (none_if_tie and len(sorted_outcomes) > 1
        and leximin_cmpt(sorted_outcomes[1], max_outcome) == 0):
        return None

    max_index = outcomes.index(max_outcome)
    max_vote = list(vote_to_outcomes.keys())[max_index]
    
    return max_vote

def make_comparator(less_than):
    def compare(x, y):
        if less_than(x, y):
            return -1
        elif less_than(y, x):
            return 1
        else:
            return 0
    return compare

def leximin_less_than(y, x):
    for xi, yi in zip(sorted(x), sorted(y)):
        if xi > yi:
            return True
        if yi > xi:
            break
    return False

def run_fair_dominated(gameState, none_if_tie=True):
    '''
    Defaulting to always `none_if_tie` = True
    '''
    modes_and_counts = []

    vote_to_outcomes = gameState.vote_to_outcomes()

    max_count = None
    max_outcome = None
    for vote in vote_to_outcomes:
        counts = Counter(vote_to_outcomes[vote])

        most_common = counts.most_common(2)
        mode, count = most_common[0]
        if len(most_common) < 2 or most_common[1][1] < count:
            # This is not a tie
            modes_and_counts.append((mode, count, vote))

    if len(modes_and_counts) == 0:
        return None

    # First sort based on count with biggest first
    modes_and_counts_sorted = \
        sorted(modes_and_counts, key=lambda x: x[1], reverse=True)
    max_count = modes_and_counts_sorted[0][1]
    max_count_only = list(filter(lambda x: x[1] == max_count, modes_and_counts_sorted))

    max_count_only_mode_sorted = sorted(max_count_only, key=lambda x: x[0], reverse=True)

    # This means there is a tie for the min element
    if (len(max_count_only_mode_sorted) > 1 and 
        max_count_only_mode_sorted[1][0] >= max_count_only_mode_sorted[0][0]):
        return None

    return max_count_only_mode_sorted[0][2]

def run_fair_dominated(gameState, none_if_tie=True):
    '''
    Defaulting to always `none_if_tie` = True
    '''
    if not gameState.equalCredences():
        raise ValueError("Have not implemented asymmetric version.")

    modes_and_counts = []

    vote_to_outcomes = gameState.vote_to_outcomes()

    max_count = None
    max_outcome = None
    for vote in vote_to_outcomes:
        counts = Counter(vote_to_outcomes[vote])

        most_common = counts.most_common(2)
        mode, count = most_common[0]
        if len(most_common) < 2 or most_common[1][1] < count:
            # This is not a tie
            modes_and_counts.append((mode, count, vote))

    if len(modes_and_counts) == 0:
        return None

    # First sort based on count with biggest first
    modes_and_counts_sorted = \
        sorted(modes_and_counts, key=lambda x: x[1], reverse=True)
    max_count = modes_and_counts_sorted[0][1]
    max_count_only = list(filter(lambda x: x[1] == max_count, modes_and_counts_sorted))

    max_count_only_mode_sorted = sorted(max_count_only, key=lambda x: x[0], reverse=True)

    # This means there is a tie for the min element
    if (len(max_count_only_mode_sorted) > 1 and 
        max_count_only_mode_sorted[1][0] >= max_count_only_mode_sorted[0][0]):
        return None

    return max_count_only_mode_sorted[0][2]

def run_theil_index(gameState, none_if_tie=True):
    '''The theil T index; https://en.wikipedia.org/wiki/Theil_index
    This is some transformation of the Atkinson index'''
    if not gameState.equalCredences():
        raise ValueError("Have not implemented asymmetric version.")

    indicies = {}

    vote_to_outcomes = gameState.vote_to_outcomes()

    for vote in vote_to_outcomes:
        mu = np.mean(vote_to_outcomes[vote])
        if mu != 0:
            n = len(vote_to_outcomes[vote])
            term_sum = 0
            for xi in vote_to_outcomes[vote]:
                frac = xi / mu
                term_sum += frac * np.log(frac)
            result = (1 / n) * term_sum
        else: 
            result = 0
        indicies[vote] = result

    max_vote = max_of_dict(indicies, none_if_tie)

    return max_vote

def run_gini(gameState, none_if_tie=True):
    '''The Gini coefficient https://en.wikipedia.org/wiki/Gini_coefficient'''
    if not gameState.equalCredences():
        raise ValueError("Have not implemented asymmetric version.")

    indicies = {}

    vote_to_outcomes = gameState.vote_to_outcomes()

    for vote in vote_to_outcomes:
        mu = np.mean(vote_to_outcomes[vote])
        n = len(vote)
        if mu != 0:
            term_sum = 0
            for xi in vote_to_outcomes[vote]:
                for xj in vote_to_outcomes[vote]:
                    term_sum += abs(xi - xj)
            denominator = (2 * (n ** 2) * mu)
            result = term_sum / denominator
            # TODO: making it negative here so we can maximize
        else:
            result = 0
        indicies[vote] = -result

    max_vote = max_of_dict(indicies, none_if_tie)

    return max_vote

def run_equality_efficiency(gameState, none_if_tie=True, alpha=.5, asymmetric=False):
    '''Inspired by Fehr Schmidt 1999 

    alpha is the preference for equality -- higher for more quality, lower for less
    equal as a default


    `asymmetric` is whether or not to use our botique way of allowing for differences in group
    sizes

     Efficiency + Equality
    \argmax_{c \in C} 
            ( \sum{a \in A} u_a(c) \times b_a ) - 
            \frac{1}{|A|} (\sum_{a \in A} \sum_{a' \in A} |u_a(c) - u_{a'}(c)| \times max(b_a / b_{a'}, b_{a'} / b_{a}) )

    '''
    if not gameState.equalCredences():
        raise ValueError("Have not implemented asymmetric version.")

    indicies = {}

    for vote in gameState.vote_agent_outcomes:

        efficiency = 0
        for agent in gameState.vote_agent_outcomes[vote]:
            efficiency += (gameState.vote_agent_outcomes[vote][agent] 
                           * gameState.getNormalizedCredence(agent))
        
        inequality = 0
        # TODO: this should not be a double loop; 
        # just need one and then the weight by the group size works better
        combos = list(itertools.combinations(gameState.getAgents(), 2))
        for agent, other in combos:
            u_agent = (gameState.vote_agent_outcomes[vote][agent])
            b_agent = gameState.getNormalizedCredence(agent)

            u_other = (gameState.vote_agent_outcomes[vote][other])
            b_other = (gameState.getNormalizedCredence(other))
            # Count the difference by the maximum ratio between the agents
            difference =  np.abs(u_agent - u_other)
            if asymmetric:
                difference *=  max(b_agent / b_other, b_other / b_agent)
            inequality += difference

        inequality /= len(combos)
        logger.debug(f'vote {vote}: inequality {inequality}, efficiency {efficiency}')
        indicies[vote] = (1 - alpha) * efficiency - (alpha) *  inequality

    max_vote = max_of_dict(indicies, none_if_tie)

    return max_vote       
