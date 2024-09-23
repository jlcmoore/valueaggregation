import copy
import numpy as np
import itertools
import logging
import sys
import itertools
import queue
logger = logging.getLogger(__name__)

class GameAction:

    def __init__(self, vote):
        self.vote = vote
    
    @staticmethod
    def fromList(the_list):
        return list(map(lambda x: GameAction(*x), the_list))

    @staticmethod
    def fromVoteList(the_list):
        return list(map(lambda x: GameAction(x), the_list))

    @staticmethod
    def toVoteList(the_list):
        return list(map(lambda x: x.vote, the_list))

    def isVoteAction(self):
        return True

    def __repr__(self):
        return f"({repr(self.vote)})"

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash((self.vote))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.vote == other.vote
        else:
            return False

class ProportionalChancesGameState:
    
    def __init__(self, agents_to_credences, vote_agent_outcomes):
        """
        Initializes the game state with the credence associated with each agent and 
        the rewards (outcomes) for each vote for each agent.
        """
        self.actions_taken = [] # Assuming that these correspond to: list(agents_to_credences.keys())
        self.agents_to_credences = agents_to_credences
        self.vote_agent_outcomes = vote_agent_outcomes
        # assert(credence.sum() == 1), should be the case but there might be a floating point error
        # TODO: normalize the credences? but might be nice not to.
        
    def getMaxAgent(self):
         return max(self.agents_to_credences, key=self.agents_to_credences.get)
        
    def getAgentVotes(self, agent):
        votes_to_outcomes = {}
        for vote in self.getVotes():
            votes_to_outcomes[vote] = self.vote_agent_outcomes[vote][agent]
        return votes_to_outcomes

    def getMaxVote(self, agent):
        # TODO: write test
        outcomes = self.getAgentVotes(agent)
        return max(outcomes, key=outcomes.get)

    def getMaxOutcome(self):
        values = set()
        for action in self.vote_agent_outcomes:
            for agent in self.vote_agent_outcomes[action]:
                values.add(self.vote_agent_outcomes[action][agent])

        return max(values)
        
    def __deepcopy__(self, memo):
        id_self = id(self)  # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(self.agents_to_credences, self.vote_agent_outcomes)
            _copy.actions_taken = copy.deepcopy(self.actions_taken)
            memo[id_self] = _copy 
        return _copy

    def getAgentUtilities(self, choice):
        '''the utilities respective to each agent for the choice'''
        return np.array([self.vote_agent_outcomes[choice][agent] for agent in self.getAgents()])

    def getUtility(self, agent, choice):
        return self.vote_agent_outcomes[choice][agent]

    def scoreGame(self):
        """
        For the given outcome (action take by the chance node), returns the known utility vector.
        """
        # may need to change this if playing with a depth greater than one
        if len(self.actions_taken) == 0:
            raise ValueError("No actions taken.")
        vote_action = self.actions_taken[-1]
        assert(vote_action.isVoteAction())
        return self.getAgentUtilities(vote_action.vote)
    
    def generateSuccessor(self, agentIndex, action):
        """
        Returns the node to follow this one, i.e. with the current agent's vote recorded.
        """
        nextState = copy.deepcopy(self)
        nextState.actions_taken.append(action)
        return nextState

    def equalCredences(self):
        return len(np.unique(self.agents_to_credences.values())) == 1
    
    def getAgent(self, agentIndex):
        return self.getAgents()[agentIndex]
    
    def getAgents(self):
        return list(self.agents_to_credences.keys())

    def getNormalizedCredence(self, agent):
        norm = sum(self.agents_to_credences.values())
        return (self.agents_to_credences[agent] / norm)

    def getAgentIndex(self, agent):
        return self.getAgents().index(agent)

    def getAgentCredence(self, agent):
        return self.agents_to_credences[agent]
    
    def getLegalActions(self, agentIndex):
        """
        Returns the available legal actions.
        """
        # This would need to change in a more complicated sort of game.
        return GameAction.fromVoteList(self.getVotes())

    def __repr__(self):
        return (f'{{"beliefs" : {repr(self.agents_to_credences)}, ' + 
               f'"outcomes" : {repr(self.vote_agent_outcomes)}, ' +
               f'"actions_taken" : {repr(self.actions_taken)}}}')

    def __str__(self):
        return (f"beliefs: {repr(self.agents_to_credences)}\n" + 
               f"outcomes: {repr(self.vote_agent_outcomes)}\n" +
               f"actions taken: {repr(self.actions_taken)}")

    def __hash__(self):
        return hash(self.__repr__())
    
    def actions_to_probabilities(self):
        """
        Based on the votes tallied so far, returns the sum of the credences of each of the agents
        that have taken each vote.
        """
        return self._actions_to_probabilities(self.actions_taken)

    def _actions_to_probabilities(self, actions):
        votes = GameAction.toVoteList(actions)
        return self._votes_to_probabilities(votes)

    def _votes_to_probabilities(self, votes):
        result = {}
        # if playing a game of a greater depth should just look at past four actions
        for vote in self.getVotes():
            result[vote] = 0
        for vote, agent in zip(votes[-self.numAgents():], self.getAgents()):
            result[vote] += self.agents_to_credences[agent]
        return result
    
    def numAgents(self):
        return len(self.agents_to_credences.keys())
    
    def getActionsTaken(self):
        return self.actions_taken[-self.numAgents():]

    def getVotesTaken(self):
        return GameAction.toVoteList(self.getActionsTaken())

    def numVotes(self):
        return len(self.vote_agent_outcomes.keys())

    def getVotes(self):
        return list(self.vote_agent_outcomes.keys())

    def getActionType(self):
        return GameAction

    def getAgentOutcomes(self, agent):
        outcomes = {}
        for vote in self.vote_agent_outcomes:
            outcomes[vote] = self.vote_agent_outcomes[vote][agent]
        return outcomes

    def payoffFromVotes(self, votes):
        """
        Provided the given votes, returns the corresponding utility vector.
        """
        return ProportionalChancesGameState.payoffFromVotes(self, votes)

    def payoffFromVotes(gameState, votes):
        # with these as the votes taken score the game
        vote_probabilities = gameState._votes_to_probabilities(votes)
        expectedValueVector = np.zeros(gameState.numAgents())

        for vote in gameState.getVotes(): # have to calculate the expectation as this is a chance node
            nextUtilityVector = gameState.getAgentUtilities(vote)
            if vote_probabilities[vote] != 0: # to deal with NaN from multiplying inf by 0
                for i in range(len(expectedValueVector)):
                    # These result is NaN otherwise
                    opposite_infinities = ((nextUtilityVector[i] == float('inf') and
                                            expectedValueVector[i] == float('-inf')) or
                                           (nextUtilityVector[i] == float('-inf') and 
                                            expectedValueVector[i] == float('inf')))
                    if not opposite_infinities:
                        expectedValueVector[i] += nextUtilityVector[i] * vote_probabilities[vote]
        return expectedValueVector

    def toArray(self):
        output = []
        for action in self.vote_agent_outcomes:
            for agent in self.vote_agent_outcomes[action]:
                output.append({'utility'  : self.vote_agent_outcomes[action][agent],
                               'agent'    : agent,
                               'credence' : self.agents_to_credences[agent],
                               'action'   : action})
        return output

    @classmethod
    def fromArray(cls, array):
        beliefs = {}
        outcomes = {}
        for item in array:
            if item['action'] not in outcomes:
                outcomes[item['action']] = {}
            outcomes[item['action']][item['agent']] = item['utility']
            beliefs[item['agent']] = item['credence']
        return cls(beliefs, outcomes)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.vote_agent_outcomes == other.vote_agent_outcomes and 
                    self.agents_to_credences == other.agents_to_credences and 
                    self.actions_taken == other.actions_taken)
        else:
            return False

    def vote_to_outcomes(self):
        '''TODO: normalize based on group size'''
        return {vote : list(agents.values()) \
            for vote, agents in self.vote_agent_outcomes.items()}

def normalizeOutcomes(gameState, norm_range):
    """
    Normalizes the outcomes linearly to the given range.
    """
    newGameState = copy.deepcopy(gameState)
    agent_ranges = {}
    for agent in newGameState.getAgents():
        outcomes = newGameState.getAgentOutcomes(agent)
        agent_ranges[agent] = (min(outcomes.values()), max(outcomes.values()))

    for vote in newGameState.vote_agent_outcomes:
        for agent in newGameState.vote_agent_outcomes[vote]:
            outcome = newGameState.vote_agent_outcomes[vote][agent]
            newGameState.vote_agent_outcomes[vote][agent] = np.interp(outcome, agent_ranges[agent],
                                                              norm_range)
    return newGameState

# could use a logistic function to deal with infinities, if desired                

def agentRangeNormalizeOutcomes(gameState):
    """
    Normalizes the outcomes linearly to the absolute value range
    of each agent, offset to the interval (1, 1 + agent_range).
    For use in Nash Bargain.
    """
    newGS = copy.deepcopy(gameState)
    agent_ranges = {}
    for agent in newGS.getAgents():
        outcomes = newGS.getAgentOutcomes(agent)
        agent_ranges[agent] = (min(outcomes.values()), max(outcomes.values()))

    def interpolate(number, agent):
        return np.interp(number, agent_ranges[agent], 
                        (1, agent_ranges[agent][1] - agent_ranges[agent][0] + 1))

    for vote in newGS.vote_agent_outcomes:
        for agent in newGS.vote_agent_outcomes[vote]:
            outcome = newGS.vote_agent_outcomes[vote][agent]
            newGS.vote_agent_outcomes[vote][agent] = interpolate(outcome, agent)
    return newGS

def bordaNormalizeOutcomes(gameState):
    # The utility of an option is the number of options worse
    # than it less than the number of options better than it 
    # $u_a^{Borda}(c) = |c \in C : u_a(c) \prec u_a(C)| - |c \in C: u_a(C) \succ u_a(C)|$
    # Uses 1-based indexing to allow for consumption in the nash equilibrium approach
    # Allows ties between ranks
    newGameState = copy.deepcopy(gameState)

    agent_queues = {}
    # Will populate this map with queues for each agent sorted by utility
    for vote in newGameState.vote_agent_outcomes:
        for agent in newGameState.vote_agent_outcomes[vote]:
            if agent not in agent_queues:
                # This is a min-heap
                agent_queues[agent] = queue.PriorityQueue(newGameState.numVotes())
            agent_queues[agent].put((newGameState.vote_agent_outcomes[vote][agent], vote))

    for agent in agent_queues:
        rank = 1               # Using 1-based indexing
        number_ranks_equal = 0 # Allows multiple votes to have the same rank
        last_priority = None
        while not agent_queues[agent].empty():

            # The smallest one comes out first and thus has the lowest value
            priority, vote = agent_queues[agent].get()

            if last_priority is not None:
                if last_priority == priority:
                    # They have the same priority, so give the same rank
                    number_ranks_equal += 1
                else:
                    # They have different priorities, so reset the counter
                    rank += number_ranks_equal + 1
                    number_ranks_equal = 0

            newGameState.vote_agent_outcomes[vote][agent] = rank

            last_priority = priority

    return newGameState

def generate_n_by_m_game(num_players, num_actions, equal_rewards=True, gameType=ProportionalChancesGameState):
    """
    Generates a game with num_players and num_actions where they all have the same reward of
    one if equal_rewards is True or reward of -1 for all actions != the number of the player
    and a reward of 1 for the action that is the players number.
    They are all given equal credence.
    """

    outcomes = {}
    for action in range(num_actions):
        action_outcomes = {}
        for player in range(num_players):
            value = 1
            if not equal_rewards and player != action:
                value = -1
            action_outcomes[player] = value
        outcomes[action] = action_outcomes

    credences = {}
    for player in range(num_players):
        credences[player] = 1/num_players

    return gameType(credences, outcomes)


def countActions(num, generator):
    '''Use this to compare the size of action space generations,
    as in NegotiationGameState.generateCoalitions and generateCoalitions'''
    count = {}
    for i in range(num):
        count[i] = {}
        for j in range(num):
            count[i][j] = len(generator(range(i), range(j)))
    return count

class NegotiationAction(GameAction):

    def __init__(self, coalition, vote):
        self.coalition = frozenset(coalition)
        self.vote = vote

    def fromList(the_list):
        return list(map(lambda x: NegotiationAction(*x), the_list))

    def fromVoteList(the_list):
        return list(map(lambda x: NegotiationAction({}, x), the_list))

    def __repr__(self):
        return f"({repr(self.coalition)}, {repr(self.vote)})"

    def isVoteAction(self):
        return len(self.coalition) == 0

    def __hash__(self):
        return hash((self.coalition, self.vote))

    def __eq__(self, other):
        return self.coalition == other.coalition and self.vote == other.vote

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.coalition == other.coalition and super().__eq__(other)
        else:
            return False

class NegotiationGameState(ProportionalChancesGameState):
    # here we are changing just the action space not the vote space

    def __init__(self, agents_to_credences, vote_agent_outcomes):
        super().__init__(agents_to_credences, vote_agent_outcomes)
        self.possible_actions = type(self).generateCoalitions(self.getVotes(), self.getAgents())

    @staticmethod
    def getActionType():
        return NegotiationAction

    def getLegalActions(self, agentIndex):
        """
        Returns the available legal actions.
        """
        actions = []
        if agentIndex < self.numAgents(): # branch node
            agent = self.getAgent(agentIndex)
            # TODO: pruning allows the first player to force all other players to 
            # do best by the first. No way around this, seemingly.
            #  filter(lambda x: agent in x.coalition, self.possible_actions)
            actions = type(self).pruneCoalitions(self.possible_actions, agent, self.actions_taken)
        else: # chance node
            actions = type(self).getActionType().fromVoteList(self.getVotes())
        # logger.debug(f"Legal actions: {actions}")
        return actions

    @classmethod
    def generateCoalitions(cls, votes, agents):
        coalition_lists = [itertools.combinations(agents,i) for i in range(1,len(agents)+1,1)]
        coalitions = itertools.chain(*coalition_lists)
        vote_coalition_combos = list(itertools.product(coalitions, votes))
        actions = [cls.getActionType()(*coalition) for coalition in vote_coalition_combos]
        return actions

    @staticmethod
    def pruneCoalitions(possible_actions, agent, actions_taken=[]):
        result = []
        # the only relevant coalitions are those in which the agent appears that do
        # not disagree with previous commitments
        for action in possible_actions:
            if agent not in action.coalition:
                continue
            broken = False
            for previous_action in actions_taken:
                if previous_action.coalition == action.coalition:
                    if previous_action.vote != action.vote:
                        broken = True
                        break
                # if coalitions overlap but are not equal
                elif previous_action.coalition & action.coalition:
                    broken = True
                    break
            if not broken:
                result.append(action)
            
        return result

def generateCoalitions(cls, votes, agents):
    # This is too big to be worth it!
    combo_lists = [itertools.combinations(votes,i) for i in range(1,len(votes)+1,1)]
    vote_combos = list(itertools.chain(*combo_lists))
    agent_vote_combos = [itertools.product(str(agent), vote_combos) for agent in agents]
    possible_conditions = list(itertools.product(*agent_vote_combos))
    return possible_conditions

class VoteGameState(ProportionalChancesGameState):

    disagreement_point = 0 # the default value for failing to come to consensus
    majority_point = .5

    def __init__(self, agents_to_credences, vote_agent_outcomes,
                 defaultPolicyFunction=None, defaultPolicyVector=False):
        """
        Expects a vector of all utilities if defaultPolicyVector is `True`
        """
        super().__init__(agents_to_credences, vote_agent_outcomes)
        self.majority_point = .5
        if defaultPolicyFunction is None:
            # the default value for failing to come to consensus:
            self.disagreement_point_vector = np.full(self.numAgents(), 0)
        else:
            if not defaultPolicyVector:
                default_vote = defaultPolicyFunction(self)
                self.disagreement_point_vector = self.getAgentUtilities(default_vote)
            else: 
                self.disagreement_point_vector = defaultPolicyFunction(self)

    def getDisagreementUtility(self, agent):
        return self.disagreement_point_vector[self.getAgentIndex(agent)]

    def __deepcopy__(self, memo):
        id_self = id(self)  # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = super().__deepcopy__(memo)
            _copy.disagreement_point_vector = self.disagreement_point_vector
            _copy.majority_point = self.majority_point
            memo[id_self] = _copy 
        return _copy

    def scoreGame(self):
        """
        For the given outcome (action take by the chance node), returns the known utility vector.
        """
        if len(self.actions_taken) == 0:
            raise ValueError("No actions taken.")

        return self.payoffFromVotes(self.getVotesTaken())   

    def payoffFromVotes(self, votes):
        """
        Provided the given votes, returns the corresponding utility vector.
        """
        votes_to_credences = self._votes_to_probabilities(votes)
        # TODO: in the future, make the voting strategy a parameter
        max_vote = max(votes_to_credences, key=votes_to_credences.get)
        max_credence = votes_to_credences[max_vote]

        if max_credence > self.majority_point:
            return self.getAgentUtilities(max_vote)
        else:
            return self.disagreement_point_vector
