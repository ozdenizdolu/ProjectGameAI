# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:10:47 2022
"""

import unittest
import itertools
import random
import logging

from ..reversi_game import Reversi

WHITE, BLACK = Reversi.WHITE, Reversi.BLACK

def count_depth(to_depth):
    stack = [(Reversi.initial_state(), to_depth)]
    count = 0
    while stack:
        current, depth = stack.pop()
        if depth == 0:
            count = count + 1
        else:
            stack.extend([
                (current.after(move, None), depth - 1)
                for move in current.moves()])
    return count

def generate_depth(to_depth, initial_state = None):
    if initial_state == None:
        initial_state = Reversi.initial_state()
    stack = [([initial_state], to_depth)]
    while stack:
        current, depth = stack.pop()
        if depth == 0:
            yield current
        else:
            stack.extend([
                (current + [current[-1].after(move, None)], depth - 1)
                for move in current[-1].moves()])
    

class ReversiTests(unittest.TestCase):
    
    def test_perft(self):
        """
        Counts the number of positions starting from the initial
        position, and compares it with the numbers on the internet.

        """
        log = logging.getLogger('ReversiTests.test_perft')
        
        
        check_to_depth = 8
        known_values = {
            1:               4,
            2:              12,
            3:              56,
            4:             244,
            5:            1396,
            6:            8200,
            7:           55092,
            8:          390216,
            9:         3005288,
            10:       24571284,
            11:      212258800,
            12:     1939886636,
            # Data below is from a less reliable source
            13:    18429641748, 
            14:   184042084512,
            15:  1891832540064,
            16: 20301186039128}
        
        
        
        for d in range(1, check_to_depth + 1):
            answer = count_depth(d)
            log.debug('depth: {} Known = {} Program = {}'.format(
                d,known_values[d], answer))
            self.assertEqual(known_values[d], answer, ' At'
                             + ' level {}'.format(d))
    
    def test_no_pieces(self):
        """
        If a player loses all of their pieces, the game should be over.
        """
        random.seed(777)
        num_of_pieces = random.choice(list(range(65)))
        raw_board = [1]*num_of_pieces + [0]*(64-num_of_pieces)
        random.shuffle(raw_board)
        
        boards = {
            WHITE: [raw_board[i:i+8] for i in range(0, len(raw_board), 8)],
            BLACK: [-1 *raw_board[i:i+8] for i in range(0, len(raw_board), 8)]
                }
        
        
        for color1, color2 in itertools.product(*(([WHITE,BLACK],)*2)):
            statement = len(Reversi.custom_state(
                boards[color1], color2).moves()) == 0
            self.assertTrue((color1 == color2) != statement)
        
    
if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "ReversiTests.test_perft" ).setLevel( logging.DEBUG )
    unittest.main()
    
    