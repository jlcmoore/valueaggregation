import logging

import unittest

class TestAssertArrays(unittest.TestCase):

    def assertArrayItemsEqual(self, one, two):
        for a, b in zip(one, two): 
            self.assertEqual(a, b)

# logging.basicConfig(level=logging.DEBUG)
