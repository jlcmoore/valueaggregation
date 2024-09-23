import unittest
from .context import TestAssertArrays
import value_aggregation as pm

class TestRunMFT(unittest.TestCase):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_one_action)
        result = pm.run_mft(gameState)
        
        self.assertEqual(result, '1')
        
        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.run_mft(gameState)
        
        self.assertEqual(result, '1')

        one_player_two_actions_tie = {'1' : {'A' : 0},
                                      '2' : {'A' : 0}}

    def test_zero_sum_two_player(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.run_mft(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.run_mft(gameState)
        self.assertEqual(result, '2')

#         Haven't set it up to deal with ties yet...
#         gameState = pm.ProportionalChancesGameState({'A' : .5, 'B' : .5}, outcomes)
#         result = pm.run_mft(gameState)
#         self.assertEqual(result, '2')
        
        gameState = pm.ProportionalChancesGameState({'A' : .6, 'B' : .4}, outcomes)
        result = pm.run_mft(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : .4, 'B' : .6}, outcomes)
        result = pm.run_mft(gameState)
        self.assertEqual(result, '2')

class TestRunMFO(unittest.TestCase):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_one_action)
        result = pm.run_mfo(gameState)
        
        self.assertEqual(result, '1')
        
        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.run_mfo(gameState)
        
        self.assertEqual(result, '1')

        one_player_two_actions_tie = {'1' : {'A' : 0},
                                      '2' : {'A' : 0}}

    def test_zero_sum_two_player(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.run_mfo(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.run_mfo(gameState)
        self.assertEqual(result, '2')

#         Haven't set it up to deal with ties yet...
#         gameState = pm.ProportionalChancesGameState({'A' : .5, 'B' : .5}, outcomes)
#         result = pm.run_mft(gameState)
#         self.assertEqual(result, '2')
        
        gameState = pm.ProportionalChancesGameState({'A' : .6, 'B' : .4}, outcomes)
        result = pm.run_mfo(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : .4, 'B' : .6}, outcomes)
        result = pm.run_mfo(gameState)
        self.assertEqual(result, '2')
    
class TestRunMEC(unittest.TestCase):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_one_action)
        result = pm.run_mec(gameState)
        
        self.assertEqual(result, '1')
        
        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.run_mec(gameState)
        
        self.assertEqual(result, '1')

        one_player_two_actions_tie = {'1' : {'A' : 0},
                                      '2' : {'A' : 0}}
        
        # TODO: should the answer be one here? If not it is a test for expectiMax, not for 
        # run_expectimax
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_two_actions_tie)
        result = pm.run_mec(gameState)
        
        self.assertEqual(result, '1')

    def test_zero_sum_two_player(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.run_mec(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.run_mec(gameState)
        self.assertEqual(result, '2')

#         Haven't set it up to deal with ties yet...
#         gameState = pm.ProportionalChancesGameState({'A' : .5, 'B' : .5}, outcomes)
#         result = pm.run_mft(gameState)
#         self.assertEqual(result, '2')
        
        gameState = pm.ProportionalChancesGameState({'A' : .6, 'B' : .4}, outcomes)
        result = pm.run_mec(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : .4, 'B' : .6}, outcomes)
        result = pm.run_mec(gameState)
        self.assertEqual(result, '2')

    def test_dominant_low_credence(self):
        action_agent_outcomes = {'1' : {'A' : 1, 'C' : -10000},
                                 '2' : {'A' : -1, 'C' : 10000}}

        gameState = pm.ProportionalChancesGameState({'A' : .99, 'C' : .01}, action_agent_outcomes)
        result = pm.run_mec(gameState)
        self.assertEqual(result, '2')

class TestRandomDictatorPolicy(TestAssertArrays):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_one_action)
        result = pm.randomDictatorPolicy(gameState)
        
        self.assertEqual(result, [0])

        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.randomDictatorPolicy(gameState)
        
        self.assertEqual(result, [1])

    def test_zero_sum_two_player(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.randomDictatorPolicy(gameState)
        self.assertArrayItemsEqual(result, [1, -1])
        
        gameState = pm.ProportionalChancesGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.randomDictatorPolicy(gameState)
        self.assertArrayItemsEqual(result, [-1, 1])

        gameState = pm.ProportionalChancesGameState({'A' : .5, 'B' : .5}, outcomes)
        result = pm.randomDictatorPolicy(gameState)
        self.assertArrayItemsEqual(result, [0, 0])

class TestRunNashBargain(TestAssertArrays):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.VoteGameState({'A' : 1}, one_player_one_action)
        result = pm.run_nash_bargain(gameState)
        
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : .001}, one_player_one_action)
        result = pm.run_nash_bargain(gameState)
        
        self.assertEqual(result, '1')
        
        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.VoteGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.run_nash_bargain(gameState)
        
        self.assertEqual(result, '1')

    def test_zero_sum_two_player(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                 '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '2')

        # is not well behaved when the utilities are 1 because 1^0 = 0

        outcomes = {'1' : {'A' : 2, 'B' : -2},
                    '2' : {'A' : -2, 'B' : 2}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '2')
        
        # TODO: these are failing
        gameState = pm.VoteGameState({'A' : .6, 'B' : .4}, outcomes)        
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.VoteGameState({'A' : .4, 'B' : .6}, outcomes)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '2')

    def test_random_dictator_disagreement(self):

        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 0}, outcomes,
                                     defaultPolicyFunction=pm.randomDictatorPolicy, 
                                     defaultPolicyVector=True)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '1')

        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : 0, 'B' : 1}, outcomes,
                                     defaultPolicyFunction=pm.randomDictatorPolicy, 
                                     defaultPolicyVector=True)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '2')

        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : .4, 'B' : .6}, outcomes,
                                     defaultPolicyFunction=pm.randomDictatorPolicy, 
                                     defaultPolicyVector=True)
        result = pm.run_nash_bargain(gameState)
        self.assertEqual(result, '2')

class TestLeximin(TestAssertArrays):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.VoteGameState({'A' : 1}, one_player_one_action)
        result = pm.run_leximin(gameState)
        
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : .001}, one_player_one_action)
        result = pm.run_leximin(gameState)
        
        self.assertEqual(result, '1')
        
        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.VoteGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.run_leximin(gameState)
        
        self.assertEqual(result, '1')

    def test_zero_pareto_two_player(self):
        outcomes = {'1' : {'A' :  1, 'B' : 1},
                    '2' : {'A' :  2, 'B' : 2}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_leximin(gameState)
        self.assertEqual(result, '2')

        outcomes = {'1' : {'A' :  1, 'B' : 1},
                    '2' : {'A' :  0, 'B' : 5},
                    '3' : {'A' :  2, 'B' : 2}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_leximin(gameState)
        self.assertEqual(result, '3')

        outcomes = {'1' : {'A' :  1, 'B' : 1},
                    '2' : {'A' :  1, 'B' : 5},
                    '3' : {'A' :  2, 'B' : 2}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_leximin(gameState)
        self.assertEqual(result, '3')

        outcomes = {'1' : {'A' :  1, 'B' : 1},
                    '2' : {'A' :  2, 'B' : 5},
                    '3' : {'A' :  3, 'B' : 3}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_leximin(gameState)
        self.assertEqual(result, '3')

        outcomes = {'1' : {'A' :  51, 'B' : 1, 'C' : 51},
                    '2' : {'A' :  1, 'B' : 1, 'C' : 51}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_leximin(gameState)
        self.assertEqual(result, '1')

        outcomes = {'1' : {'A' :  1, 'B' : 1, 'C' : 1},
                    '2' : {'A' :  51, 'B' : 51, 'C' : 51}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_leximin(gameState)
        self.assertEqual(result, '2')

class TestFairDominated(TestAssertArrays):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.VoteGameState({'A' : 1}, one_player_one_action)
        result = pm.run_fair_dominated(gameState)
        
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : .001}, one_player_one_action)
        result = pm.run_fair_dominated(gameState)
        
        self.assertEqual(result, '1')
        
    def test_zero_pareto_two_player(self):
        outcomes = {'1' : {'A' :  1, 'B' : 1},
                    '2' : {'A' :  2, 'B' : 2}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_fair_dominated(gameState)
        self.assertEqual(result, '2')

        outcomes = {'1' : {'A' :  1, 'B' : 1},
                    '2' : {'A' :  0, 'B' : 5},
                    '3' : {'A' :  2, 'B' : 2}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_fair_dominated(gameState)
        self.assertEqual(result, '3')

        outcomes = {'1' : {'A' :  1, 'B' : 1},
                    '2' : {'A' :  0, 'B' : 5},
                    '3' : {'A' :  2, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_fair_dominated(gameState)
        self.assertEqual(result, '1')

        outcomes = {'1' : {'A' :  1, 'B' : 1, 'C' : 2},
                    '2' : {'A' :  0, 'B' : 5, 'C' : 1},
                    '3' : {'A' :  2, 'B' : 1, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_fair_dominated(gameState)
        self.assertIsNone(result)

        outcomes = {'1' : {'A' :  1, 'B' : 1, 'C' : 2},
                    '2' : {'A' :  0, 'B' : 5, 'C' : 1},
                    '3' : {'A' :  2, 'B' : 3, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_fair_dominated(gameState)
        self.assertEqual(result, '1')

        outcomes = {'1' : {'A' :  51, 'B' : 51, 'C' : 101},
                    '2' : {'A' :  1, 'B' : 1, 'C' : 51},
                    '3' : {'A' :  1, 'B' : 1, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_fair_dominated(gameState)
        self.assertEqual(result, '3')

        outcomes = {'1' : {'A' :  101, 'B' : 1, 'C' : 101},
                    '2' : {'A' :  51, 'B' : 1, 'C' : 51},
                    '3' : {'A' :  1, 'B' : 51, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_fair_dominated(gameState)
        self.assertEqual(result, '1')

class TestTheilIndex(TestAssertArrays):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.VoteGameState({'A' : 1}, one_player_one_action)
        result = pm.run_theil_index(gameState)
        
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : .001}, one_player_one_action)
        result = pm.run_theil_index(gameState)
        
        self.assertEqual(result, '1')

    def test_multi_player(self):

        outcomes = {'1' : {'A' :  101, 'B' : 1, 'C' : 101},
                    '2' : {'A' :  1, 'B' : 101, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_theil_index(gameState)
        self.assertEqual(result, '1')

        outcomes = {'1' : {'A' :  101, 'B' : 1, 'C' : 101},
                    '2' : {'A' :  51, 'B' : 1, 'C' : 51},
                    '3' : {'A' :  1, 'B' : 51, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_theil_index(gameState)
        self.assertEqual(result, '2')

class TestGiniIndex(TestAssertArrays):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.VoteGameState({'A' : 1}, one_player_one_action)
        result = pm.run_gini(gameState)
        
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : .001}, one_player_one_action)
        result = pm.run_gini(gameState)
        
        self.assertEqual(result, '1')

    def test_multi_player(self):

        outcomes = {'1' : {'A' :  101, 'B' : 1, 'C' : 101},
                    '2' : {'A' :  1, 'B' : 101, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_gini(gameState)
        self.assertEqual(result, '1')

        outcomes = {'1' : {'A' :  101, 'B' : 1, 'C' : 101},
                    '2' : {'A' :  51, 'B' : 1, 'C' : 51},
                    '3' : {'A' :  1, 'B' : 51, 'C' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 1}, outcomes)
        result = pm.run_gini(gameState)
        self.assertEqual(result, '2')


if __name__ == '__main__':
    unittest.main()
