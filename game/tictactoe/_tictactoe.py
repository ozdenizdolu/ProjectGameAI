"""
This module provides the game logic of the game TicTacToe. 
"""

import itertools
import random

import numpy as np
from numpy.random import default_rng

np_rng = default_rng()


class TicTacToe:
    X, O, empty = 100,101,102
    OUTCOME = 1462198
    
    _name_dict = {X:'X', O: 'O', empty: '_'}
    
    players = [X,O]
    
    @classmethod
    def _other_player(cls, player):
        assert player in {TicTacToe.X, TicTacToe.O}
        return TicTacToe.X if player == TicTacToe.O else TicTacToe.O

    @classmethod
    def initial_state(cls):
        return TicTacToeState(np.array([TicTacToe.empty]*9).reshape(3,3), TicTacToe.X)
    
    @classmethod
    def random_state(cls):
        mapping = {0: TicTacToe.X, 1: TicTacToe.O, 2: TicTacToe.empty}
        while True:
            random_board = np_rng.integers(3, size=(3,3))
            for i in range(3):
                random_board[random_board == i] = mapping[i]
            random_turn = random.choice([TicTacToe.X, TicTacToe.O])
            state = TicTacToeState(random_board, random_turn)
            #TODO
            if not state.is_game_over():
                return state
    
    @classmethod
    def all_states(cls):
        states = []
        
        for raw_board, turn in itertools.product(
                itertools.product([TicTacToe.empty, TicTacToe.X, TicTacToe.O],
                                  repeat = 9),
                [TicTacToe.X, TicTacToe.O]):
            board = np.array(raw_board).reshape(3,3)
            new_state = TicTacToeState(board, turn)
            states.append(new_state)
        
        return states
            
            

class TicTacToeState:
    
    
    def __init__(self, board, turn):
        """Initialize to the starting position."""
        self._board = board
        self._turn = turn
    
    def outcomes(self, move, pick_one = False):
        if not pick_one:
            return [TicTacToe.OUTCOME], [1]
        else:
            return TicTacToe.OUTCOME
    
    def _look_for_strike(self):
        # Returns the first strike it detects.
        # BEWARE! This function returns the value of the constant
        # X or O. It would compare false under is comparison.
        # The returned object comes from numpy's array
        # when it is assigned by [..] = (0,1,X) it gets the string
        # 'X', not the object one. So the returned object is not X
        # but a copy of it. One should be careful with constant values
        # with primitive values. The problem would not occur if
        # X was an object instead, i.e. X = object().
        
        rows = (list(itertools.product([i], [0,1,2])) for i in [0,1,2])
        columns = (list(itertools.product([0,1,2], [i])) for i in [0,1,2])
        diagonals = [list(zip([0,1,2],[0,1,2])), list(zip([0,1,2],[2,1,0]))]
        
        for strike in itertools.chain(rows, columns, diagonals):
            
            strike_contains = set(self._board[ind] for ind in strike)
            if len(strike_contains) == 1 and not (TicTacToe.empty in strike_contains):
                return strike, strike_contains.pop()
        
        return None
        
    
    def game_final_evaluation(self):
        try:
            strike, value = self._look_for_strike()
        except TypeError:
            #No strike so it must be a draw (if it is game over of course)
            return {TicTacToe.X: 0., TicTacToe.O: 0.}
        
        if value == TicTacToe.X:
            return {TicTacToe.X: 1., TicTacToe.O: -1.}
        elif value == TicTacToe.O:
            return {TicTacToe.X: -1., TicTacToe.O: 1.}
        else:
            assert False
                
        
    def game(self):
        return TicTacToe
    
    def turn(self):
        return self._turn

    def moves(self):
        return list(zip(
            *((self._board == TicTacToe.empty).nonzero()),
            itertools.repeat(self._turn)))

    def after(self, move, outcome):
        if (outcome != TicTacToe.OUTCOME) and (outcome is not None):
            raise ValueError('Outcome is not recognized.')
        
        # Make sure that the move is syntactically correct
        if not len(move) == 3:
            raise ValueError('Move is not recognized.')
        new_board = np.array(self._board, copy = True)
        new_board[move[0], move[1]] = move[2]
        return TicTacToeState(new_board, TicTacToe._other_player(self._turn))
        
    
    def is_game_over(self):
        # The game may be over due to not being able to play
        # Another reason is that one player has already won
        # If there are no strike and there are moves then
        # the game is not over.
        if len(self.moves()) == 0:
            return True
        
        if self._look_for_strike() is not None:
            return True
        
        return False
            
    def __str__(self):
        return ((  '{} {} {}\n'
                + '{} {} {}\n'
                + '{} {} {}\n'
                ).format(*[TicTacToe._name_dict[self._board[indice]]
                         for indice in np.ndindex(self._board.shape)])
                + 'turn: {}'.format(TicTacToe._name_dict[self._turn]))
    
    def __eq__(self, other):
        if not isinstance(other, TicTacToeState):
            return False
        else:
            return ((self._board == other._board).all()
                    and self._turn == other._turn)
    
    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            self._cached_hash = hash(
                (hash(self._board.tobytes()), hash(self._turn)))
            return self._cached_hash
