import unittest
import numpy as np
import copy
from .context import TestAssertArrays
import value_aggregation as pm

class TestGameNodesProportionalChances(TestAssertArrays):

    def test_generate_n_by_m_game(self):

        utilities, actions = pm.expectiMax(pm.generate_n_by_m_game(1, 1), 1, 0)
        votes = pm.GameAction.toVoteList(actions)
        self.assertEqual(votes, [0])
        self.assertArrayItemsEqual(utilities, np.array([1]))

        utilities, actions = pm.expectiMax(pm.generate_n_by_m_game(2, 1), 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([1, 1]))
        votes = pm.GameAction.toVoteList(actions)
        self.assertEqual(votes, [0, 0])

        utilities, actions = pm.expectiMax(pm.generate_n_by_m_game(2, 2, False), 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([0, 0]))
        votes = pm.GameAction.toVoteList(actions)
        self.assertEqual(votes, [0, 1])

        utilities, actions = pm.expectiMax(pm.generate_n_by_m_game(2, 2), 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([1, 1]))
        votes = pm.GameAction.toVoteList(actions)
        self.assertEqual(votes, [0, 0])
    
    def test_expectiMax_zero_depth(self):
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        with self.assertRaises(ValueError):
            utilities, actions = pm.expectiMax(gameState, 0, 0)
        
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        answer = [pm.GameAction('1')]
        gameState.actions_taken.append(answer[0])
        utilities, actions = pm.expectiMax(gameState, 0, 1)
        self.assertEqual(actions, answer)
        self.assertEqual(utilities[0], 1)
        
        gameState = pm.ProportionalChancesGameState({'A' : 0}, outcomes)
        answer = [pm.GameAction('1')]
        gameState.actions_taken.append(answer[0])
        utilities, actions = pm.expectiMax(gameState, 0, 0)
        self.assertEqual(actions, answer)
        self.assertEqual(utilities[0], 1)
        
    def test_expectiMax_one_depth(self):
        outcomes = {'1' : {'A' : float('inf')}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertEqual(actions, [pm.GameAction('1')])
        self.assertEqual(utilities[0], float('inf'))
        
        outcomes = {'1' : {'A' : float('-inf')}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertEqual(actions, [pm.GameAction('1')])
        self.assertEqual(utilities[0], float('-inf'))

        # assert votes empty and utilities as passed, also for other values
        # do this

        outcomes = {'1' : {'A' : float('inf')}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([float('inf')]))
        self.assertEqual(actions, [pm.GameAction('1')])
        
        outcomes = {'1' : {'A' : float('-inf')}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([float('-inf')]))
        self.assertEqual(actions, [pm.GameAction('1')])
        
        outcomes = {'1' : {'A' : float('-inf')}, '2' : {'A' : float('inf')}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([float('inf')]))
        self.assertEqual(actions, [pm.GameAction('2')])

    def test_expectiMax_one_depth_numbers(self):
        
        # test with numbers not strings
        outcomes = {1 : {1 : 1}}
        gameState = pm.ProportionalChancesGameState({1 : 0}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([0]))
        self.assertEqual(actions, [pm.GameAction(1)])

        # test different confidences
        outcomes = {1 : {1 : 1}}
        gameState = pm.ProportionalChancesGameState({1 : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([1]))
        self.assertEqual(actions, [pm.GameAction(1)])

        outcomes = {1 : {1 : 1}}
        gameState = pm.ProportionalChancesGameState({1 : .1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertArrayItemsEqual(utilities, np.array([.1]))
        self.assertEqual(actions, [pm.GameAction(1)])

    def test_expectiChance_zero_depth(self):
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        answer = [pm.GameAction('1')]
        gameState.actions_taken.append(answer[0])
        utilities, actions = pm.expectiChance(gameState, 0, 1)
        self.assertEqual(actions, answer)
        self.assertEqual(utilities[0], 1)
        
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        gameState.actions_taken.append(pm.GameAction('1'))
        with self.assertRaises(ValueError):
            utilities, actions = pm.expectiChance(gameState, 0, 0)

        gameState = pm.ProportionalChancesGameState({'A' : 0}, outcomes)
        answer = [pm.GameAction('1')]
        gameState.actions_taken.append(answer[0])
        utilities, actions = pm.expectiChance(gameState, 0, 1)
        self.assertEqual(actions, answer)
        self.assertEqual(utilities[0], 1)

        outcomes = {'1' : {'A' : float('inf')}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        answer = [pm.GameAction('1')]
        gameState.actions_taken.append(answer[0])
        utilities, actions = pm.expectiChance(gameState, 0, 1)
        self.assertEqual(actions, answer)
        self.assertEqual(utilities[0], float('inf'))
        
        outcomes = {'1' : {'A' : float('-inf')}}
        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        answer = [pm.GameAction('1')]
        gameState.actions_taken.append(answer[0])
        utilities, actions = pm.expectiChance(gameState, 0, 1)
        self.assertEqual(actions, answer)
        self.assertEqual(utilities[0], float('-inf'))
        
    def test_expectiChance_one_depth(self):
        
        outcomes = {'1' : {'A' : float('inf')}, '2' : {'A': float('-inf')}}

        gameState = pm.ProportionalChancesGameState({'A' : 1}, outcomes)
        answer = [pm.GameAction('1')]
        gameState.actions_taken.append(answer[0])
        utilities, actions = pm.expectiChance(gameState, 1, 1)
        self.assertEqual(actions, answer)
        self.assertEqual(utilities[0], float('inf'))
        
        # This is to test for the NaN summing case
        outcomes = {'1' : {'A' : 0, 'B' : float('inf')},
                    '2' : {'A' : 0, 'B' : float('-inf')}}

        gameState = pm.ProportionalChancesGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        answer = [pm.GameAction('1'), pm.GameAction('2')]
        gameState.actions_taken = copy.deepcopy(answer)
        utilities, actions = pm.expectiChance(gameState, 1, 2)
        self.assertEqual(actions, answer)
        self.assertArrayItemsEqual(utilities, np.array([0., float('inf')]))

class TestGameNodesNegotiation(TestAssertArrays):

    def test_expectiMax_zero_depth(self):
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.NegotiationGameState({'A' : 1}, outcomes)
        with self.assertRaises(ValueError):
            utilities, actions = pm.expectiMax(gameState, 0, 0)
        
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.NegotiationGameState({'A' : 1}, outcomes)
        gameState.actions_taken = [pm.NegotiationAction({}, '1')]
        utilities, actions = pm.expectiMax(gameState, 0, 1)
        self.assertEqual(actions, [pm.NegotiationAction({},'1')])
        self.assertEqual(utilities[0], 1)
        
        gameState = pm.NegotiationGameState({'A' : 0}, outcomes)
        gameState.actions_taken.append(pm.NegotiationAction({},'1'))
        utilities, actions = pm.expectiMax(gameState, 0, 0)
        self.assertEqual(actions, [pm.NegotiationAction({},'1')])
        self.assertEqual(utilities[0], 1)

    def test_expectiChance_zero_depth(self):
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.NegotiationGameState({'A' : 1}, outcomes)
        gameState.actions_taken.append(pm.NegotiationAction({},'1'))
        utilities, actions = pm.expectiChance(gameState, 0, 1)
        self.assertEqual(actions, [pm.NegotiationAction({},'1')])
        self.assertEqual(utilities[0], 1)
        
        outcomes = {'1' : {'A' : 1}}
        gameState = pm.NegotiationGameState({'A' : 1}, outcomes)
        gameState.actions_taken.append(pm.NegotiationAction({},'1'))
        with self.assertRaises(ValueError):
            utilities, actions = pm.expectiChance(gameState, 0, 0)

        gameState = pm.NegotiationGameState({'A' : 0}, outcomes)
        gameState.actions_taken.append(pm.NegotiationAction({},'1'))
        utilities, actions = pm.expectiChance(gameState, 0, 1)
        self.assertEqual(actions, [pm.NegotiationAction({},'1')])
        self.assertEqual(utilities[0], 1)

    def test_expectiChance_one_depth(self):
        
        outcomes = {'1' : {'A' : 1}, '2' : {'A': -1}}

        gameState = pm.NegotiationGameState({'A' : 1}, outcomes)
        gameState.actions_taken.append(pm.NegotiationAction({'A'},'1'))
        utilities, actions = pm.expectiChance(gameState, 1, 1)
        self.assertEqual(actions, [pm.NegotiationAction({'A'},'1')])
        self.assertEqual(utilities[0], 1)
        
        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.NegotiationGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        answer = [pm.NegotiationAction({'A'},'1'), pm.NegotiationAction({'B'},'2')]
        gameState.actions_taken = copy.copy(answer)
        utilities, actions = pm.expectiChance(gameState, 1, 2)
        self.assertEqual(actions, answer)
        self.assertArrayItemsEqual(utilities, np.array([0, 0]))

        gameState = pm.NegotiationGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        answer = [pm.NegotiationAction({'A','B'},'1'), pm.NegotiationAction({'B'},'1')]
        gameState.actions_taken = copy.copy(answer)
        utilities, actions = pm.expectiChance(gameState, 1, 2)
        self.assertEqual(actions, answer)
        self.assertArrayItemsEqual(utilities, np.array([1, -1]))

    def test_expectiMax_recursive(self):

        outcomes = {'1' : {'A' : 1}, '2' : {'A': -1}}

        gameState = pm.NegotiationGameState({'A' : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertEqual(actions, [pm.NegotiationAction({'A'},'1')])
        self.assertEqual(utilities[0], 1)

        outcomes = {'1' : {'A' : 1, 'B' : 1}}

        gameState = pm.NegotiationGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        self.assertEqual(actions, [pm.NegotiationAction({'A'},'1'), pm.NegotiationAction({'B'},'1')])
        self.assertArrayItemsEqual(utilities, np.array([1, 1]))

        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.NegotiationGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        answer = [pm.NegotiationAction({'A'},'1'), pm.NegotiationAction({'B'},'2')]
        utilities, actions = pm.expectiMax(gameState, 1, 0)
        # this is known to be in error
        # self.assertEqual(actions, answer)
        # self.assertArrayItemsEqual(utilities, np.array([0, 0]))

class TestGameNodesVote(TestAssertArrays):

    def test_expectiMax_recursive(self):
        outcomes = {'1' : {'A' : 1}, '2' : {'A': -1}}

        gameState = pm.VoteGameState({'A' : 1}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0, False)
        self.assertEqual(actions, [pm.GameAction('1')])
        self.assertEqual(utilities[0], 1)

        outcomes = {'1' : {'A' : 1, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        utilities, actions = pm.expectiMax(gameState, 1, 0, False)
        self.assertEqual(actions, [pm.GameAction('1'), pm.GameAction('1')])
        self.assertArrayItemsEqual(utilities, np.array([1, 1]))

        outcomes = {'1' : {'A' : 1, 'B' : -1},
                    '2' : {'A' : -1, 'B' : 1}}

        gameState = pm.VoteGameState({'A' : 1/2, 'B' : 1/2}, outcomes)
        answer = [pm.GameAction('1'), pm.GameAction('2')]
        utilities, actions = pm.expectiMax(gameState, 1, 0, False)
        self.assertEqual(actions, answer)
        self.assertArrayItemsEqual(utilities, np.array([0, 0]))

if __name__ == '__main__':
    unittest.main()
