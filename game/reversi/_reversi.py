"""
Created on Sat Jun 11 00:54:52 2022

@author: Özdeniz Dolu
"""

import random

# import numpy as 

# __all__ = ['OUTCOME', 'PASS_MOVE', 'WHITE', 'BLACK', 'initial_state',
           # 'ReversiGameState']

class Reversi:
    """
    This is a class object containing the rules of the game.
    """
    
    OUTCOME =  'OUTCOME'
    PASS_MOVE = 'PASS_MOVE'
    
    WHITE = 1
    BLACK = -1
    
    players = [WHITE, BLACK]
    
    @classmethod
    def initial_state(cls):
        board = tuple(tuple(
            (1 if (i + j) % 2 == 0 else -1) if (2 < i < 5 and 2 < j < 5) else 0
            for j in range(8)) for i in range(8))
        turn = Reversi.BLACK
        return ReversiGameState(board, turn)
    
    @classmethod
    def custom_state(cls, board, turn, last_player_passed = False):
        """Board should be 8x8 nested lists."""
        if len(board) != 8:
            raise ValueError()
            
        for row in board:
            if len(row) != 8:
                raise ValueError()
        
        return ReversiGameState(board, turn, last_player_passed=last_player_passed)


#
#
# =============== INTEGRATE THESE INTO THE INTERFACE ==================


#DONE

# =====================================================================
#
#

_directions = [(i, j) for j in range(-1, 2) for i in range(-1, 2)]
_directions.remove((0, 0))


def _direction_square_generator(board, square, direction):
    i,j = square
    del_i, del_j = direction
    while 0 <= i < 8 and 0 <= j < 8:
        yield (i, j, board[i][j])
        i = i + del_i
        j = j + del_j
        
def _look_for_match(board, square, direction, my_colour):
    gen = _direction_square_generator(board, square, direction)
    next(gen)
    return_list = []
    while True:
        try:
            return_list.append(next(gen))
        except StopIteration:
            return
        if return_list[-1][-1] == my_colour and len(return_list) != 0:
            return return_list
        elif return_list[-1][-1] == 0:
            return None

class ReversiGameState:
    
    # self._board is a tuple of tuples. Each item is a row.
    
    
    #after (move, outcome) return gamestate
    # is_game_over
    # moves
    # game_final_evaluation()  returns evaluation for each player, if game not over raise exception   
    # turn return turn 
    # outcomes(move) : give all possible outcomes tuple [outcomes],[prob of outcomes]
    def __init__(self, board, turn, last_player_passed = False):
        self._board = board
        self._turn = turn
        self._last_player_passed = last_player_passed

    # This function is integrated to the moves for performance
    # def _find_move_in_direction(self, position, direction):
    #     i , j = position
    #     di, dj = direction
    #     enemy_found = False
    #     empty_found = False
    #     i = i + di
    #     j = j + dj
    #     while i < 8 and j < 8 and i > -1 and j > -1:
    #         if not enemy_found:
    #             if self._board[i][j] == -self._turn:
    #                 enemy_found = True 
    #             else:
    #                 return False
    #         elif not empty_found:
    #             if self._board[i][j] == 0:
    #                 return (i, j)
    #             elif self._board[i][j] == self._turn:
    #                 return False
    #         i = i + di
    #         j = j + dj
            
    def outcomes(self, move, pick_one = False):
        if not pick_one:
            return [Reversi.OUTCOME], [1]
        else:
            return Reversi.OUTCOME
        
    def game_final_evaluation(self):
        """
        Assuming the game is over, it returns -1, 0, or 1
        depending on the result of the game.
        
        Shouldn't be called when is_game_over() is False.
        """
        white_count = sum((1 for row in self._board for square_colour in row \
                           if square_colour == 1))
        black_count = sum((1 for row in self._board for square_colour in row \
                           if square_colour == -1))
        
        if white_count > black_count:
            return {Reversi.WHITE: 1, Reversi.BLACK: -1}
        elif black_count > white_count:
            return {Reversi.WHITE: -1, Reversi.BLACK: 1}
        else:
            return {Reversi.WHITE: 0, Reversi.BLACK: 0}
        
    def game(self):
        return Reversi
    
    def turn(self):
        return self._turn

    def moves(self):
        
        valid_moves = []
        
        for i in range(8):
            for j in range(8):
                if self._board[i][j] == self._turn:
                    for direction in _directions:
                        #find move in direction
                        
                        di, dj = direction
                        enemy_found = False
                        empty_found = False
                        i_ = i + di
                        j_ = j + dj
                        move = None
                        while i_ < 8 and j_ < 8 and i_ > -1 and j_ > -1:
                            if not enemy_found:
                                if self._board[i_][j_] == -self._turn:
                                    enemy_found = True 
                                else:
                                    break
                            elif not empty_found:
                                if self._board[i_][j_] == 0:
                                    move = (i_, j_)
                                    break
                                elif self._board[i_][j_] == self._turn:
                                    break
                            i_ = i_ + di
                            j_ = j_ + dj
                        
                        #found move (or not)
                        
                        if move is not None:
                            valid_moves.append(move)
                            
        if len(valid_moves) == 0:
            if self._last_player_passed:
                return []
            else:
                return [Reversi.PASS_MOVE]
        else:
            #method creates duplicate moves so convert to set.
            return list(set(valid_moves))
    

    def after(self, move, outcome):
        if (outcome != Reversi.OUTCOME) and (outcome is not None):
            raise ValueError('Outcome is not recognized.')
        
        board_list_copy = [list(row) for row in self._board]
        if not move == Reversi.PASS_MOVE:
            matches = []
            for direction in _directions:
                match = _look_for_match(self._board, move, direction, self._turn)
                if match is not None:
                    matches.append(match)
            for match in matches:
                for square in match:
                    board_list_copy[square[0]][square[1]] = self._turn
            board_list_copy[move[0]][move[1]] = self._turn
        board_copy = tuple(tuple(row) for row in board_list_copy)
        return ReversiGameState(board_copy, self._turn * -1,
                                last_player_passed = (move == Reversi.PASS_MOVE))
    
    def is_game_over(self):
        return len(self.moves()) == 0
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        imaging = {Reversi.BLACK: 'X', Reversi.WHITE: 'O', 0: '-'}
        return ''.join(
            [("{: >3} "*8)[:-1].format(*[imaging[square] for square in row])
             +'\n' for row in self._board]
            + ['\nTurn: {}'.format(imaging[self.turn()])]
            + ['\nDid last player pass {}'.format(self._last_player_passed)])

    def __eq__(self, other):
        #TODO check if self._board == ... does the thing expected.
        if not isinstance(other, ReversiGameState):
            return False
        else:
            return (self._board == other._board
                and self._turn == other._turn
                and self._last_player_passed == other._last_player_passed)
        
    def __hash__(self):
        return hash((self._board, self._turn, self._last_player_passed))
