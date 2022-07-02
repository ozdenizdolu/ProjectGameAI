"""
Provides functions for translating game related information to a form
which neural networks can understand.
"""

import torch
import numpy as np

from ._tictactoe import TicTacToe

def state_translator(state):
    # # Make 2 feature planes for stones
    XS = torch.tensor(state._board == TicTacToe.X,
                      dtype=torch.float).reshape((1,3,3))
    OS = torch.tensor(state._board == TicTacToe.O, 
                      dtype=torch.float).reshape((1,3,3))
    
    player_plane = (torch.ones(state._board.shape)
                    if state._turn == TicTacToe.X 
                    else torch.zeros(state._board.shape)).reshape(1,3,3)
    
    return torch.cat([XS,OS, player_plane], dim = 0).reshape((27,))

def move_translator(move_distribution):
    # move_distribution is given in dict format
    # first do it with a numpy array to avoid getting in the way of 
    # autograd. 
    temp = np.zeros((3,3))
    for move, probability in move_distribution.items():
        i, j, piece = move
        temp[i,j] = probability
        
    return torch.tensor(temp, dtype=torch.float).reshape((9,))

def inv_move_translator(state, legal_moves, output):
    output = output.reshape(3,3)
    # Do not forget to put zero probability on illegal moves.
    temp = {move: output[move[0],move[1]].item()
            for move in legal_moves}
    normalization_constant = sum(value for value in temp.values())
    if normalization_constant <= 0:
        raise ValueError("No probability on valid moves.")
    
    return {move: value/normalization_constant for move, value in temp.items()}
    
def evaluation_translator(evaluation):
    #mapping to tensor
    return torch.tensor([evaluation[TicTacToe.X]], dtype=torch.float)

def inv_evaluation_translator(tensor):
    return {TicTacToe.X: tensor.item(), TicTacToe.O: -tensor.item()}

