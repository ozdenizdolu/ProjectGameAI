"""
Development Note
----------------

Neural network + translator = evaluator

"""

import torch

import game.tictactoe.translator_1 as tr

class StandardTicTacToeTranslator:
    
    def states_to_tensor(self, states, device):
        """
        Returns a tensor of valid network inputs from a list of game states (sent to the same device as the network)

        Parameters
        ----------
        states : list
            A list of TicTacToeState's.

        Returns
        -------
        tensor
            A two dimensional tensor whose 0th dimension is the same size as len(states)
        """
        return torch.cat([tr.state_as_flattened_feature_planes(state).unsqueeze(0) for state in states],
                         dim=0).to(device)
    
    # Converts the list of move_distributions to the same type as 
    # the neural network output move_distributions
    def dists_to_tensor(self, move_distributions, device):
        return torch.cat([tr.dist_to_tensor(move_dist).unsqueeze(0) for move_dist in move_distributions],
                        dim=0).to(device)
    
    # Converts the list of evaluations to the same type as 
    # the neural network output evaluations
    def evals_to_tensor(self, states, evaluations, device):
        return torch.cat([tr.eval_tensor_wrt_X(evaluation).unsqueeze(0) for evaluation in evaluations],
                        dim=0).to(device)

    def tensor_to_dists(self, states, legal_moves_list, moves_tensor):
        """
        Returns a list of move distribution dictionaries from the given move distributions tensor. 
        Move distributions tensor is assumed to be the output of the network.

        Parameters
        ----------
        states : list
            A list of TicTacToeState's. Each state in this list is associated with the corresponding move distribution in the moves_tensor.
        legal_moves_list : list
            A list of list of legal moves for each state in states.
        moves_tensor: tensor
            A two dimensional tensor where each move distribution is indexed by the 0th dimension.

        Returns
        -------
        list
            A list of move distribution dictionaries.
        """
        if moves_tensor.ndim != 2:
            raise ValueError("moves_tensor should have 2 dimensions")
        if len(states) != len(legal_moves_list):
            raise ValueError("len(states) should be equal to len(legal_moves_list")
        if len(states) != moves_tensor.shape[0]:
            raise ValueError("len(states) should be equal to size of 0th dimension of moves_tensor")
        return [tr.tensor_to_dist(states[i], legal_moves_list[i], moves_tensor[i])
                for i in range(len(states))]

    # Converts the evaluation output of the neural network 
    # to be compatible with the rest of the system
    def tensor_to_evals(self, states, evals_tensor):
        if evals_tensor.ndim != 2:
            raise ValueError("evals_tensor should have 2 dimensions")
        return [tr.get_eval_wrt_X(evals_tensor[i]) for i in range(evals_tensor.shape[0])]

    def translate_data(self, data, device):
        states, dists, evals = zip(*data)
        return (self.states_to_tensor(states, device),
                self.dists_to_tensor(dists, device),
                self.evals_to_tensor(states, evals, device))

class TokenLikeTicTacToeTranslator:
    
    def states_to_tensor(self, states, device):
        # This one will translate wrt the current player.
        # There are many states
        return torch.cat(
            [tr.state_as_ordered_tokens(state,device).unsqueeze(dim=0)
             for state in states], dim = 0)
        
    
    # Converts the list of move_distributions to the same type as 
    # the neural network output move_distributions
    def dists_to_tensor(self, move_distributions, device):
        return torch.cat([tr.dist_to_tensor(move_dist).unsqueeze(0) for move_dist in move_distributions],
                        dim=0).to(device)
    
    # Converts the list of evaluations to the same type as 
    # the neural network output evaluations
    def evals_to_tensor(self, states, evaluations, device):
        return torch.cat(
            [torch.tensor(evaluation[state.turn()], device=device
                         ).reshape((1,1))
             for state, evaluation in zip(states, evaluations)],
            dim = 0)

    def tensor_to_dists(self, states, legal_moves_list, moves_tensor):
        """
        Returns a list of move distribution dictionaries from the given move distributions tensor. 
        Move distributions tensor is assumed to be the output of the network.

        Parameters
        ----------
        states : list
            A list of TicTacToeState's. Each state in this list is associated with the corresponding move distribution in the moves_tensor.
        legal_moves_list : list
            A list of list of legal moves for each state in states.
        moves_tensor: tensor
            A two dimensional tensor where each move distribution is indexed by the 0th dimension.

        Returns
        -------
        list
            A list of move distribution dictionaries.
        """
        if moves_tensor.ndim != 2:
            raise ValueError("moves_tensor should have 2 dimensions")
        if len(states) != len(legal_moves_list):
            raise ValueError("len(states) should be equal to len(legal_moves_list")
        if len(states) != moves_tensor.shape[0]:
            raise ValueError("len(states) should be equal to size of 0th dimension of moves_tensor")
        return [tr.tensor_to_dist(states[i], legal_moves_list[i], moves_tensor[i])
                for i in range(len(states))]

    # Converts the evaluation output of the neural network 
    # to be compatible with the rest of the system
    def tensor_to_evals(self, states, evals_tensor):
        return [tr.get_eval_wrt_current_player(state, evals_tensor[i,:])
                for state, i in zip(states,range(evals_tensor.shape[0]))]

    def translate_data(self, data, device):
        states, dists, evals = zip(*data)
        return (self.states_to_tensor(states, device),
                self.dists_to_tensor(dists, device),
                self.evals_to_tensor(states, evals, device))
    
#TODO    
# class StandardReversiTranslator:
    
#     def __init__(self):
#         pass
    
#     def states_to_tensor(self, states, device):
#         """
#         Returns a tensor of valid network inputs from a list of game states (sent to the same device as the network)

#         Parameters
#         ----------
#         states : list
#             A list of TicTacToeState's.

#         Returns
#         -------
#         tensor
#             A two dimensional tensor whose 0th dimension is the same size as len(states)
#         """
        
#         return torch.cat(#TODO).to(device)
    
#     # Converts the list of move_distributions to the same type as 
#     # the neural network output move_distributions
#     def dists_to_tensor(self, move_distributions, device):
#         return torch.cat([tr.dist_to_tensor(move_dist).unsqueeze(0) for move_dist in move_distributions],
#                         dim=0).to(device)
    
#     # Converts the list of evaluations to the same type as 
#     # the neural network output evaluations
#     def evals_to_tensor(self, states, evaluations, device):
#         return torch.cat([tr.eval_tensor_wrt_X(evaluation).unsqueeze(0) for evaluation in evaluations],
#                         dim=0).to(device)

#     def tensor_to_dists(self, states, legal_moves_list, moves_tensor):
#         """
#         Returns a list of move distribution dictionaries from the given move distributions tensor. 
#         Move distributions tensor is assumed to be the output of the network.

#         Parameters
#         ----------
#         states : list
#             A list of TicTacToeState's. Each state in this list is associated with the corresponding move distribution in the moves_tensor.
#         legal_moves_list : list
#             A list of list of legal moves for each state in states.
#         moves_tensor: tensor
#             A two dimensional tensor where each move distribution is indexed by the 0th dimension.

#         Returns
#         -------
#         list
#             A list of move distribution dictionaries.
#         """
#         if moves_tensor.ndim != 2:
#             raise ValueError("moves_tensor should have 2 dimensions")
#         if len(states) != len(legal_moves_list):
#             raise ValueError("len(states) should be equal to len(legal_moves_list")
#         if len(states) != moves_tensor.shape[0]:
#             raise ValueError("len(states) should be equal to size of 0th dimension of moves_tensor")
#         return [tr.tensor_to_dist(states[i], legal_moves_list[i], moves_tensor[i])
#                 for i in range(len(states))]

#     # Converts the evaluation output of the neural network 
#     # to be compatible with the rest of the system
#     def tensor_to_evals(self, states, evals_tensor):
#         if evals_tensor.ndim != 2:
#             raise ValueError("evals_tensor should have 2 dimensions")
#         return [tr.get_eval_wrt_X(evals_tensor[i]) for i in range(evals_tensor.shape[0])]

#     def translate_data(self, data, device):
#         states, dists, evals = zip(*data)
#         return (self.states_to_tensor(states, device),
#                 self.dists_to_tensor(dists, device),
#                 self.evals_to_tensor(states, evals, device))