import torch

from ._reversi import Reversi

#TODO I FORGOT PASS MOVE!!

def state_as_two_flattened_planes_and_player_plane(state):
    board = torch.tensor(state._board)
    white_plane = torch.where(board == Reversi.WHITE,
                    torch.ones_like(board, dtype=torch.float),
                    torch.zeros_like(board, dtype=torch.float)).flatten()
    black_plane = torch.where(board == Reversi.BLACK,
                    torch.ones_like(board, dtype=torch.float),
                    torch.zeros_like(board, dtype=torch.float)).flatten()
    # Like in AlphaGoZero paper
    if not state.turn() in [Reversi.WHITE, Reversi.BLACK]:
        raise ValueError("State is not a reversi state.")
    player_color = 1. if state.turn() == Reversi.WHITE else 0.
    player_plane = (torch.ones_like(board, dtype=torch.float) * player_color
                    ).flatten()
    
    return torch.cat([white_plane, black_plane, player_plane])

def dist_to_tensor(dist):
    #distribution is a mapping of moves to scalars.
    tensor = torch.zeros(8,8, dtype=torch.float)
    for move, probability in dist.items():
        if move != Reversi.PASS_MOVE:
            tensor[move[0], move[1]] = probability
        
    try:
        legal_move_weight = dist[Reversi.PASS_MOVE]
    except KeyError:
        legal_move_weight = 0
        
    return torch.cat([tensor.flatten(),
            torch.tensor(legal_move_weight, dtype=torch.float).reshape(1)])

def eval_to_tensor_wrt_white(evaluation):
    return torch.tensor(evaluation[Reversi.WHITE], dtype=torch.float)

def tensor_to_dist(state, legal_moves, move_tensor):
    if move_tensor.shape != (65,):
        raise ValueError('64 + Pass move is assumed!')
    
    board_tensor = move_tensor[:64].reshape(8,8)
    
    weight_on_legals = {move: board_tensor[move[0], move[1]].item()
                        for move in legal_moves if move != Reversi.PASS_MOVE}
    
    if Reversi.PASS_MOVE in legal_moves:
        weight_on_legals[Reversi.PASS_MOVE] = move_tensor[-1].item()
    
    normalization_constant = sum(value for value in weight_on_legals.values())
    
    if normalization_constant <= 0:
        raise ValueError('''The sum of probabilities 
                         of legal moves is {}. The number of legal moves is {}.
                         state.is_game_over() returns {}
                         '''.format(normalization_constant,
                         len(legal_moves), state.is_game_over()))
    
    return {move: value/normalization_constant
            for move, value in weight_on_legals.items()}
    
    
def tensor_to_eval_wrt_white(eval_tensor):
    return {Reversi.WHITE: eval_tensor.item(),
            Reversi.BLACK: -eval_tensor.item()}