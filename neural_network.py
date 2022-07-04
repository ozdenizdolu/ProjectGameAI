"""

"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import game.tictactoe.translator_1 as tr

class TicTacToe_defaultNN(nn.Module):
    
    def __init__(self, current_device):
        super(TicTacToe_defaultNN, self).__init__()
        self._current_device = current_device
        self.layer1 = nn.Sequential(
            nn.Linear(27, 100),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.Linear(100, 9),
            nn.Softmax(dim = 1),
        )
        self.layer3_2 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x_moves = self.layer3_1(x)
        x_eval = self.layer3_2(x)
        return x_moves, x_eval
    
    def calculate_loss(self, data, dist_loss_fn=None, eval_loss_fn=None):
        """
        data is a triplet (x, dist_target, eval_target)
        """
        if dist_loss_fn is None:
            dist_loss_fn = torch.nn.functional.cross_entropy
        if eval_loss_fn is None:
            eval_loss_fn = torch.nn.functional.mse_loss
        
        x, dist_target, eval_target = data
        
        dist_net, eval_net = self(x)
        
        return (dist_loss_fn(dist_net, dist_target),
                eval_loss_fn(eval_net, eval_target))
        
        
        
        

    # Gets a list of game states and returns a batch of valid inputs to network, sent to the same device 
    def states_to_tensor(self, states):
        return torch.cat([tr.state_to_tensor(state).unsqueeze(0) for state in states],
                         dim=0).to(self._current_device)
    
    # Converts the move distribution output of the neural network
    # to be compatible with the rest of the system
    def tensor_to_dists(self, states, legal_moves_list, moves_tensor):
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
    def tensor_to_evals(self, evals_tensor):
        if evals_tensor.ndim != 2:
            raise ValueError("evals_tensor should have 2 dimensions")
        return [tr.tensor_to_eval(evals_tensor[i]) for i in range(evals_tensor.shape[0])]
    
    # Converts the list of move_distributions to the same type as 
    # the neural network output move_distributions
    def dists_to_tensor(self, move_distributions, device = None):
        if device == None:
            device = self._current_device
        if type(move_distributions) != type([]):
            raise TypeError("move_distributions should be a list")
        return torch.cat([tr.dist_to_tensor(move_dist).unsqueeze(0) for move_dist in move_distributions],
                        dim=0).to(device)
    
    # Converts the list of evaluations to the same type as 
    # the neural network output evaluations
    def evals_to_tensor(self, evaluations, device = None):
        if device == None:
            device = self._current_device
        if type(evaluations) != type([]):
            raise TypeError("evaluations should be a list")
        return torch.cat([tr.eval_to_tensor(evaluation).unsqueeze(0) for evaluation in evaluations],
                        dim=0).to(device)
    
    # A helper function that is compatible with the evaluator protocol of the system
    def as_evaluator(self):
        def evaluator(state, legal_moves, player):
            x = tr.state_to_tensor(state).unsqueeze(dim=0)
            move_dist, eval = self.forward(x)
            return (tr.tensor_to_dist(move_dist.squeeze(0)), 
                    tr.tensor_to_eval(eval.squeeze(0)))
        return evaluator


class TicTacToe_residualNN(nn.Module):
    
    def __init__(self, current_device):
        super(TicTacToe_residualNN, self).__init__()
        self._current_device = current_device
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(27, 27)
        self.layer2 = nn.Linear(27, 27)
        self.layer3_1 = nn.Sequential(
            nn.Linear(27, 9),
            nn.Softmax(dim = 1)
        )
        self.layer3_2 = nn.Sequential(
            nn.Linear(27, 1),
            nn.Tanh(),
        )
    def forward(self, x):
        x_1 = self.layer1(x)
        x_1 = self.relu(x_1)
        x_2 = self.layer2(x_1) + x
        x_2 = self.relu(x_2)
        x_moves = self.layer3_1(x_2)
        x_eval = self.layer3_2(x_2)
        return x_moves, x_eval

    def calculate_loss(self, data, dist_loss_fn=None, eval_loss_fn=None):
        """
        data is a triplet (x, dist_target, eval_target)
        """
        if dist_loss_fn is None:
            dist_loss_fn = torch.nn.functional.cross_entropy
        if eval_loss_fn is None:
            eval_loss_fn = torch.nn.functional.mse_loss
        
        x, dist_target, eval_target = data
        
        dist_net, eval_net = self(x)
        
        return (dist_loss_fn(dist_net, dist_target),
                eval_loss_fn(eval_net, eval_target))
                
    # Gets a list of game states and returns a batch of valid inputs to network, sent to the same device 
    def states_to_tensor(self, states):
        return torch.cat([tr.state_to_tensor(state).unsqueeze(0) for state in states],
                         dim=0).to(self._current_device)
    
    # Converts the move distribution output of the neural network
    # to be compatible with the rest of the system
    def tensor_to_dists(self, states, legal_moves_list, moves_tensor):
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
    def tensor_to_evals(self, evals_tensor):
        if evals_tensor.ndim != 2:
            raise ValueError("evals_tensor should have 2 dimensions")
        return [tr.tensor_to_eval(evals_tensor[i]) for i in range(evals_tensor.shape[0])]
    
    # Converts the list of move_distributions to the same type as 
    # the neural network output move_distributions
    def dists_to_tensor(self, move_distributions, device = None):
        if device == None:
            device = self._current_device
        if type(move_distributions) != type([]):
            raise TypeError("move_distributions should be a list")
        return torch.cat([tr.dist_to_tensor(move_dist).unsqueeze(0) for move_dist in move_distributions],
                        dim=0).to(device)
    
    # Converts the list of evaluations to the same type as 
    # the neural network output evaluations
    def evals_to_tensor(self, evaluations, device = None):
        if device == None:
            device = self._current_device
        if type(evaluations) != type([]):
            raise TypeError("evaluations should be a list")
        return torch.cat([tr.eval_to_tensor(evaluation).unsqueeze(0) for evaluation in evaluations],
                        dim=0).to(device)
    
    # A helper function that is compatible with the evaluator protocol of the system
    def as_evaluator(self):
        def evaluator(state, legal_moves, player):
            x = tr.state_to_tensor(state).unsqueeze(dim=0)
            move_dist, eval = self.forward(x)
            return (tr.tensor_to_dist(move_dist.squeeze(0)), 
                    tr.tensor_to_eval(eval.squeeze(0)))
        return evaluator
