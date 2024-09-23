import unittest
import value_aggregation as pm

class TestProportionalChancesGameState(unittest.TestCase):
    
    def test_init(self):        
        # initially ProportionalChancesGameState .actions_taken should be empty
        # credences should sum to one
        pass
        
    def test_copy(self):
        # editing the state variables of a copy does not change them
        pass

    def test_scoreGame(self):
        # set-up a game with a few turns taken, see if the right score is given
        pass
     
    def test_generateSuccessor(self):
        pass
    
    def test_getAgent(self):
        pass
    
    def test_getAgents(self):
        pass
    
    def test_getLegalActions(self):
        pass
    
    def test_actions_to_probabilities(self):
        pass
    
    def test_actions_to_probabilities(self):
        pass
    
    def test_numAgents(self):
        pass
    
    def test_getActionsTaken(self):
        pass

    def test_getNormalizedCredence(self):

        gs = pm.ProportionalChancesGameState({'A' : 1}, {})
        self.assertEqual(gs.getNormalizedCredence('A'), 1)

        gs = pm.ProportionalChancesGameState({'A' : 1, 'B' : 1}, {})
        self.assertEqual(gs.getNormalizedCredence('B'), 1/2)

        gs = pm.ProportionalChancesGameState({'A' : .5, 'B' : .5}, {})
        self.assertEqual(gs.getNormalizedCredence('B'), 1/2)

    def test_varianceNormalizeOutcomes(self):

        norm_range = (-1, +1)
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.normalizeOutcomes(gameState, norm_range)

        expected_outcomes = {'1' : {'A' : 1.0, 'B' : -1.0},
                             '2' : {'A' : -1.0, 'B' : 1.0}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)
        
        outcomes = {'1' : {'A' : 1, 'B' : 0},
                    '2' : {'A' : 0, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.normalizeOutcomes(gameState, norm_range)

        expected_outcomes = {'1' : {'A' : 1.0, 'B' : -1.0},
                             '2' : {'A' : -1.0, 'B' : 1.0}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

        # does not work with infinities

        outcomes = {
        '1' : {'A': +1000,  'B': -1},
        '2' : {'A': +0   ,  'B': +0},
        '3' : {'A': -1000,  'B': +1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.normalizeOutcomes(gameState, norm_range)

        expected_outcomes = {'1' : {'A' : 1.0, 'B' : -1.0},
                             '2' : {'A' : 0.0, 'B' : 0.0},
                             '3' : {'A' : -1.0, 'B' : 1.0}}

        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

    def test_agentRangeNormalizeOutcomes(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.agentRangeNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 3, 'B' : 1},
                             '2' : {'A' : 1, 'B' : 3}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)
        
        outcomes = {'1' : {'A' : 1, 'B' : 0},
                    '2' : {'A' : 0, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.agentRangeNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 2, 'B' : 1},
                             '2' : {'A' : 1, 'B' : 2}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

        # should work with ties
        outcomes = {'1' : {'A' : 1, 'B' : 2},
                    '2' : {'A' : 1, 'B' : 2}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.agentRangeNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 1, 'B' : 1},
                             '2' : {'A' : 1, 'B' : 1}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

        outcomes = {'1' : {'A' : -10, 'B' : 1},
                    '2' : {'A' : 10, 'B' : 21}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.agentRangeNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 1, 'B' : 1},
                             '2' : {'A' : 21, 'B' : 21}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

        outcomes = {'1' : {'A' : -100, 'B' : 1},
                    '2' : {'A' : 10, 'B' : 21}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.agentRangeNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 1, 'B' : 1},
                             '2' : {'A' : 111, 'B' : 21}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

        outcomes = {
        '1'   : {'A': +60, 'B': +60},
        '2'   : {'A': +10, 'B': +100}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.agentRangeNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 51, 'B' : 1},
                             '2' : {'A' :  1, 'B' : 41}}

        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

    def test_bordaNormalizeOutcomes(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.bordaNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 2, 'B' : 1},
                             '2' : {'A' : 1, 'B' : 2}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)
        
        outcomes = {'1' : {'A' : 1, 'B' : 0},
                    '2' : {'A' : 0, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.bordaNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 2, 'B' : 1},
                             '2' : {'A' : 1, 'B' : 2}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

        outcomes = {'1' : {'A' : -9, 'B' : 999},
                    '2' : {'A' : 234.1, 'B' : 0},
                    '3' : {'A' : float('inf'), 'B' : float('-inf')}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.bordaNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 1, 'B' : 3},
                             '2' : {'A' : 2, 'B' : 2},
                             '3' : {'A' : 3, 'B' : 1}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

        # does not work with infinities 

        # should work with ties
        outcomes = {'1' : {'A' : 1, 'B' : 2},
                    '2' : {'A' : 1, 'B' : 2}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        gameState = pm.bordaNormalizeOutcomes(gameState)

        expected_outcomes = {'1' : {'A' : 1, 'B' : 1},
                             '2' : {'A' : 1, 'B' : 1}}
        self.assertEqual(gameState.vote_agent_outcomes, expected_outcomes)

    def test_toArray(self):
        outcomes = {'1' : {'A' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        result = gameState.toArray()
        expected = [{'utility' : 1, 'action' : '1','agent' : 'A', 'credence' : 1}]
        self.assertEqual(result, expected)

        # TODO: perhaps should normalize these credences?
        gameState = pm.ProportionalChancesGameState({'A' : 10}, outcomes)
        result = gameState.toArray()
        expected = [{'utility' : 1, 'action' : '1','agent' : 'A', 'credence' : 10}]
        self.assertEqual(result, expected)

        outcomes = {'1' : {'A' : 1, 'B' : 3}}

        gameState = pm.ProportionalChancesGameState({'A' : 10, 'B' : 20}, outcomes)
        result = gameState.toArray()
        expected = [{'utility' : 1, 'action' : '1','agent' : 'A', 'credence' : 10},
                    {'utility' : 3, 'action' : '1','agent' : 'B', 'credence' : 20}]
        self.assertEqual(result, expected)

    def test_fromArray(self):
        outcomes = {'1' : {'A' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        result = pm.ProportionalChancesGameState.fromArray(gameState.toArray())
        self.assertEqual(result, gameState)

        # TODO: perhaps should normalize these credences?
        gameState = pm.ProportionalChancesGameState({'A' : 10}, outcomes)
        result = pm.ProportionalChancesGameState.fromArray(gameState.toArray())
        self.assertEqual(result, gameState)

        outcomes = {'1' : {'A' : 1, 'B' : 3}}

        gameState = pm.ProportionalChancesGameState({'A' : 10, 'B' : 20}, outcomes)
        result = pm.ProportionalChancesGameState.fromArray(gameState.toArray())
        self.assertEqual(result, gameState)

class TestNegotiationGameState(unittest.TestCase):

    def test_generateCoalitions(self):
        actions = pm.NegotiationGameState.generateCoalitions(['A'], range(1))
        self.assertEqual(actions, pm.NegotiationAction.fromList([({0}, 'A')]))

        actions = pm.NegotiationGameState.generateCoalitions(['A'], range(2))
        self.assertEqual(actions, pm.NegotiationAction.fromList([({0}, 'A'), ({1}, 'A'), ({0, 1}, 'A')]))

        actions = pm.NegotiationGameState.generateCoalitions(['A', 'B'], range(2))
        self.assertEqual(actions, pm.NegotiationAction.fromList(([({0}, 'A'), ({0}, 'B'), ({1}, 'A'), ({1}, 'B'), ({0, 1}, 'A'), ({0, 1}, 'B')])))

    def test_pruneCoalitions(self):
        actions = pm.NegotiationGameState.generateCoalitions(['A'], range(1))
        pruned = pm.NegotiationGameState.pruneCoalitions(actions, 0)
        self.assertEqual(pruned, pm.NegotiationAction.fromList([({0}, 'A')]))

        actions = pm.NegotiationGameState.generateCoalitions(['A'], range(1))
        pruned = pm.NegotiationGameState.pruneCoalitions(actions, 1)
        self.assertEqual(pruned, [])

        actions = pm.NegotiationGameState.generateCoalitions(['A'], range(1))
        pruned = pm.NegotiationGameState.pruneCoalitions(actions, 1, [])
        self.assertEqual(pruned, [])

        actions = pm.NegotiationGameState.generateCoalitions(['A'], range(2))
        pruned = pm.NegotiationGameState.pruneCoalitions(actions, 1, [])
        self.assertEqual(pruned, pm.NegotiationAction.fromList([({1}, 'A'), ({0, 1}, 'A')]))

        actions = pm.NegotiationGameState.generateCoalitions(['A'], range(2))
        pruned = pm.NegotiationGameState.pruneCoalitions(actions, 1, pm.NegotiationAction.fromList([({0, 1}, 'A')]))
        self.assertEqual(pruned, pm.NegotiationAction.fromList([({0, 1}, 'A')]))

        actions = pm.NegotiationGameState.generateCoalitions(['A', 'B'], range(2))
        pruned = pm.NegotiationGameState.pruneCoalitions(actions, 1, [])
        self.assertEqual(pruned, pm.NegotiationAction.fromList([({1}, 'A'), ({1}, 'B'), ({0, 1}, 'A'), ({0, 1}, 'B')]))

        pruned = pm.NegotiationGameState.pruneCoalitions(actions, 1, pm.NegotiationAction.fromList([({0, 1}, 'A')]))
        self.assertEqual(pruned, pm.NegotiationAction.fromList([({0, 1}, 'A')]))

class TestVoteGameState(unittest.TestCase):

    pass

if __name__ == '__main__':
    unittest.main()
