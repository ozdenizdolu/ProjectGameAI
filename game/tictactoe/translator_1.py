"""
Provides functions for translating game related information to a form
which neural networks can understand.
"""

import itertools

import torch
import numpy as np

from ._tictactoe import TicTacToe


def state_as_flattened_feature_planes(state):
    # # Make 2 feature planes for stones
    XS = torch.tensor(state._board == TicTacToe.X,
                      dtype=torch.float).reshape((1,3,3))
    OS = torch.tensor(state._board == TicTacToe.O, 
                      dtype=torch.float).reshape((1,3,3))
    
    player_plane = (torch.ones(state._board.shape)
                    if state._turn == TicTacToe.X 
                    else torch.zeros(state._board.shape)).reshape(1,3,3)
    
    return torch.cat([XS,OS, player_plane], dim = 0).reshape((27,))

def dist_to_tensor(move_distribution):
    # move_distribution is given in dict format
    # first do it with a numpy array to avoid getting in the way of 
    # autograd. 
    temp = np.zeros((3,3))
    for move, probability in move_distribution.items():
        i, j, piece = move
        temp[i,j] = probability
        
    return torch.tensor(temp, dtype=torch.float).reshape((9,))

def tensor_to_dist(state, legal_moves, move_tensor):
    move_tensor = move_tensor.reshape(3,3)
    # Do not forget to put zero probability on illegal moves.
    temp = {move: move_tensor[move[0],move[1]].item()
            for move in legal_moves}
    normalization_constant = sum(value for value in temp.values())
    if normalization_constant <= 0:
        raise ValueError('''The sum of probabilities 
                         of legal moves is {}. The number of legal moves is {}.
                         state.is_game_over() returns {}
                         '''.format(normalization_constant,
                         len(legal_moves), state.is_game_over()))
    
    return {move: value/normalization_constant for move, value in temp.items()}
    
def eval_tensor_wrt_X(evaluation):
    #mapping to tensor
    return torch.tensor([evaluation[TicTacToe.X]], dtype=torch.float)

def get_eval_wrt_X(eval_tensor):
    return {TicTacToe.X: eval_tensor.item(), TicTacToe.O: -eval_tensor.item()}

def get_eval_wrt_current_player(state, eval_tensor):
    return {state.turn(): eval_tensor.item(),
            TicTacToe._other_player(state.turn()): -eval_tensor.item()}
