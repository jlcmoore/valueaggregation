import unittest
from .context import TestAssertArrays
import value_aggregation as pm
import random

class TestUtils(TestAssertArrays):

    def test_split_by(self):
        source = [1, 2, 3, 4, 5]
        expected = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 1], [5, 1, 2]]
        result = pm.split_by(source, 3)
        self.assertArrayItemsEqual(result, expected)

    def test_generate_hits(self):
        beliefs = {'A': 30, 'B': 10, 'C' : 20}

        outcomes = {
        'one'   : {'A': +1000, 'B': +1000,  'C' : 1000},
        'two'   : {'A': +1, 'B': +1, 'C' : 1},
        'three' : {'A': +1, 'B': +1,  'C' : 10000}}

        gameState = pm.VoteGameState(beliefs, outcomes)
        hits = pm.generate_hits([gameState, gameState], scenarios_per_hit=1,
                sample_size=2)

        class DictComparator(tuple):
            def __lt__(self, other):
                return self[0]['json'] < other[0]['json']
        
        self.assertEqual(hits[0][0]['mec'], 'three')
        self.assertEqual(hits[0][0]['nbs'], 'one')

    def test_encode_decode_gameState(self):
        expected = pm.VoteGameState({'A' : 1}, {'one' : {'A' : 1}})
        result = pm.decode_gameState(pm.encode_gameState(expected))
        self.assertEqual(expected, result)

        beliefs = {'A': 30, 'B': 10, 'C' : 20}

        outcomes = {
        'one'   : {'A': +1000, 'B': +1000,  'C' : 1000},
        'two'   : {'A': +1, 'B': +1, 'C' : 1},
        'three' : {'A': +1, 'B': +1,  'C' : 10000}}

        gameState = pm.VoteGameState(beliefs, outcomes)
        result = pm.decode_gameState(pm.encode_gameState(expected))
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
