"""



"""

from abc import ABC, abstractmethod

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
            nn.Softmax(0),
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

    # Converts the game state object to input of neural network
    def state_to_input(self, state):
        return tr.state_translator(state).to(self._current_device)
    
    # Converts the output of the neural network to be compatible 
    # with the rest of the system
    def inv_translate_out(self, state, legal_moves, output):
        move, ev = output
        return (
            tr.inv_move_translator(state, legal_moves, move), 
            tr.inv_evaluation_translator(ev)
        )
    
    # Converts the prior probability distributions and evaluations
    # coming from the system to the same types as the neural network output
    def translate_out(self, move_distribution, evaluation):
        return (
            tr.move_translator(move_distribution), 
            tr.evaluation_translator(evaluation)
        )
    
    # A helper function that is compatible with the evaluator protocol of the system
    def as_evaluator(self):
        def evaluator(state, legal_moves, player):
            x = self.state_to_input(state)
            out = self.forward(x)
            return self.inv_translate_out(state, legal_moves, out)
        return evaluator


class TicTacToe_residualNN(nn.Module):
    
    def __init__(self, current_device):
        super(TicTacToe_residualNN, self).__init__()
        self._current_device = current_device
        self.layer1 = nn.Sequential(
            nn.Linear(27, 27),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(27, 27),
            nn.ReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.Linear(27, 9),
            nn.Softmax(0),
        )
        self.layer3_2 = nn.Sequential(
            nn.Linear(27, 1),
            nn.Tanh(),
        )
    def forward(self, x):
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1 + x)
        x_moves = self.layer3_1(x_2 + x)
        x_eval = self.layer3_2(x_2 + x)
        return x_moves, x_eval

    # Converts the game state object to input of neural network
    def state_to_input(self, state):
        return tr.state_translator(state).to(self._current_device)
    
    # Converts the output of the neural network to be compatible 
    # with the rest of the system
    def inv_translate_out(self, state, legal_moves, output):
        move, ev = output
        return (
            tr.inv_move_translator(state, legal_moves, move), 
            tr.inv_evaluation_translator(ev)
        )
    
    # Converts the prior probability distributions and evaluations
    # coming from the system to the same types as the neural network output
    def translate_out(self, move_distribution, evaluation):
        return (
            tr.move_translator(move_distribution), 
            tr.evaluation_translator(evaluation)
        )
    
    # A helper function that is compatible with the evaluator protocol of the system
    def as_evaluator(self):
        def evaluator(state, legal_moves, player):
            x = self.state_to_input(state)
            out = self.forward(x)
            return self.inv_translate_out(state, legal_moves, out)
        return evaluator