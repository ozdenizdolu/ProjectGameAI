"""

"""

import numpy as np
import itertools

X, O, empty = 'X', 'O', '_'
OUTCOME = 1462198

def _other_player(player):
    assert player in {X,O}
    return X if player == O else O

# class TicTacToe:
#     #TODO
#     pass

def initial_state():
    return TicTacToeState(np.array([empty]*9).reshape(3,3), X)

class TicTacToeState:
    
    def __init__(self, board, turn):
        """Initialize to the starting position."""
        self._board = board
        self._turn = turn
    
    def outcomes(self, move, pick_one = False):
        if not pick_one:
            return [OUTCOME], [1]
        else:
            return OUTCOME
    
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
            if len(strike_contains) == 1 and not (empty in strike_contains):
                return strike, strike_contains.pop()
        
        return None
        
    
    def game_final_evaluation(self):
        try:
            strike, value = self._look_for_strike()
        except TypeError:
            #No strike so it must be a draw (if it is game over of course)
            return {X:0., O:0.}
        
        if value == X:
            return {X:1., O:-1.}
        elif value == O:
            return {X:-1., O:1.}
        else:
            assert False
                
        
    def game(self):
        return TicTacToe
    
    def turn(self):
        return self._turn

    def moves(self):
        return list(zip(
            *((self._board == empty).nonzero()),
            itertools.repeat(self._turn)))

    def after(self, move, outcome):
        if (outcome != OUTCOME) and (outcome is not None):
            raise ValueError('Outcome is not recognized.')
        
        # Make sure that the move is syntactically correct
        if not len(move) == 3:
            raise ValueError('Move is not recognized.')
        new_board = np.array(self._board, copy = True)
        new_board[move[0], move[1]] = move[2]
        return TicTacToeState(new_board, _other_player(self._turn))
        
    
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
        return str(self._board)

