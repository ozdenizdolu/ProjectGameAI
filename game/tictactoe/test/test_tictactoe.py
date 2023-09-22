"""

"""
import unittest
import itertools
import random
import logging

import numpy as np

from .. import TicTacToe
from .. import translator_1


class TicTacToeTests(unittest.TestCase):
    
    def test_general(self):
        random.seed(1252134)
        state = TicTacToe.initial_state()
        for i in range(9,0,-1):
            self.assertEqual(len(state.moves()), i)
            self.assertTrue(i not in {7,8,9} or not state.is_game_over())
            state = state.after(random.choice(list(state.moves())), None)
            if state.is_game_over():
                break
        
    
    def test_hash_related(self):
        random.seed(2412512)
        a = TicTacToe.initial_state()
        b = TicTacToe.initial_state()
        self.assertTrue(a is not b)
        self.assertTrue(a == b)
        _dict = {}
        try:
            _dict[a] = 5
            _dict[b] = 7
        except TypeError:
            self.fail('Problem being a key')
        self.assertEqual(len(_dict), 1)
        self.assertTrue(a in _dict.keys())
        self.assertEqual(_dict[a], 7)
        
        a = a.after(random.choice(list(a.moves())), None)
        _dict[a] = 4
        
        self.assertEqual(len(_dict), 2)
            
        
