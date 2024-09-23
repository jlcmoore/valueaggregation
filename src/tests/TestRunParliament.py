import unittest
import sys
import value_aggregation as pm

class TestRunParliament(unittest.TestCase):
    
    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_one_action)
        result = pm.run_expectimax(gameState)
        
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : 0}, one_player_one_action)
        result = pm.run_expectimax(gameState)
        
        self.assertIsNone(result)
        
        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.run_expectimax(gameState)
        
        self.assertEqual(result, '1')

        one_player_two_actions_tie = {'1' : {'A' : 0},
                                      '2' : {'A' : 0}}
        
        # TODO: should the answer be one here? If not it is a test for expectiMax, not for 
        # pm.run_expectimax
        gameState = pm.ProportionalChancesGameState({'A' : 1}, one_player_two_actions_tie)
        result = pm.run_expectimax(gameState)
        
        self.assertEqual(result, '1')

    def test_zero_sum_two_player(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '2')

        gameState = pm.ProportionalChancesGameState({'A' : .5, 'B' : .5}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)
        
        gameState = pm.ProportionalChancesGameState({'A' : .6, 'B' : .4}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.ProportionalChancesGameState({'A' : .4, 'B' : .6}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '2')
        
    def test_collaborative_two_player(self):
        outcomes = {'1' : {'A' : -1, 'B' : -1},
                    '2' : {'A' : 1, 'B' : 1}}

        gameState = pm.ProportionalChancesGameState({'A' : .5, 'B' : .5}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '2')
        
    def test_condorcet(self):
        outcomes = {'1' : {'A' : -1, 'B' : 0, 'C' : 1},
                    '2' : {'A' : 0, 'B' : 1, 'C' : -1},
                    '3' : {'A' : 1, 'B' : -1, 'C' : 0}}
        
        gameState = pm.ProportionalChancesGameState({'A' : 1/3, 'B' : 1/3, 'C' : 1/3}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)
    
        # this is just the same as above
        result = pm.run_expectimax(pm.generate_n_by_m_game(3, 3, False))
        self.assertIsNone(result)
        
        gameState = pm.ProportionalChancesGameState({'A' : 1/4, 'B' : 1/4, 'C' : 1/2}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)

        # max int game
        outcomes = {'1' : {'A' : -sys.maxsize, 'B' : 0, 'C' : sys.maxsize},
                    '2' : {'A' : 0, 'B' : sys.maxsize, 'C' : -sys.maxsize},
                    '3' : {'A' : sys.maxsize, 'B' : -sys.maxsize, 'C' : 0}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/3, 'B' : 1/3, 'C' : 1/3}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)
        
        # infinitesimal game
        outcomes = {'1' : {'A' : -1/sys.maxsize, 'B' : 0, 'C' : 1/sys.maxsize},
                    '2' : {'A' : 0, 'B' : 1/sys.maxsize, 'C' : -1/sys.maxsize},
                    '3' : {'A' : 1/sys.maxsize, 'B' : -1/sys.maxsize, 'C' : 0}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/3, 'B' : 1/3, 'C' : 1/3}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)
        
        # Any more than five will take awhile
        result = pm.run_expectimax(pm.generate_n_by_m_game(5, 5, False))
        self.assertIsNone(result)

    def test_collaborative_three_player_two_action(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1, 'C' : 1},
                    '2' : {'A' : -1, 'B' : 1, 'C': 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/3, 'B' : 1/3, 'C' : 1/3}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '1')
        
        outcomes = {'1' : {'A' : 1, 'B' : -1, 'C' : -1},
                    '2' : {'A' : -1, 'B' : 1, 'C': 1}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/3, 'B' : 1/3, 'C' : 1/3}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '2')

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/4, 'C' : 1/4}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)

    def test_collaborative_case_fails(self):
        # this is the undesirable case

        action_agent_outcomes = {'1' : {'A' : 1, 'B' : -1, 'C' : -1},
                                 '2' : {'A' : -1, 'B' : -1, 'C' : 1},
                                 '3' : {'A' : -1, 'B' : 1, 'C' : -1},
                                 '4' : {'A' : -1, 'B' : .9, 'C' : .9}}

        gameState = pm.ProportionalChancesGameState({'A' : .4, 'B' : .3, 'C' : .3}, action_agent_outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)

    # need to add test mfo, mec, mft

class TestRunParliamentVote(unittest.TestCase):

    def test_single_player(self):
        one_player_one_action = {'1' : {'A' : 0}}
        gameState = pm.VoteGameState({'A' : 1}, one_player_one_action)
        result = pm.run_expectimax(gameState)
        
        self.assertEqual(result, '1')

        gameState = pm.VoteGameState({'A' : 0}, one_player_one_action)
        result = pm.run_expectimax(gameState)
        
        self.assertIsNone(result)
        
        one_player_two_actions_split = {'1' : {'A' : 1},
                                        '2' : {'A' : -1}}
        
        gameState = pm.VoteGameState({'A' : 1}, one_player_two_actions_split)
        result = pm.run_expectimax(gameState)
        
        self.assertEqual(result, '1')

    def test_collaborative_case(self):
        # this is the undesirable case

        action_agent_outcomes = {'1' : {'A' : 1, 'B' : -1, 'C' : -1},
                                 '2' : {'A' : -1, 'B' : -1, 'C' : 1},
                                 '3' : {'A' : -1, 'B' : 1, 'C' : -1},
                                 '4' : {'A' : -1, 'B' : .9, 'C' : .9}}

        gameState = pm.VoteGameState({'A' : .4, 'B' : .3, 'C' : .3}, action_agent_outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '4')

    def test_zero_sum_two_player(self):
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : 1, 'B' : 0}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.VoteGameState({'A' : 0, 'B' : 1}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '2')

        gameState = pm.VoteGameState({'A' : .5, 'B' : .5}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertIsNone(result)
        
        gameState = pm.VoteGameState({'A' : .6, 'B' : .4}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '1')
        
        gameState = pm.VoteGameState({'A' : .4, 'B' : .6}, outcomes)
        result = pm.run_expectimax(gameState)
        self.assertEqual(result, '2')

if __name__ == '__main__':
    unittest.main()
