"""
Development note:
    
Refactoring will be needed at some point.
current_device parameter seems to be obsolete.
divide translator from the nn
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class TicTacToe_defaultNN(nn.Module):
    
    def __init__(self):
        super(TicTacToe_defaultNN, self).__init__()
        
        self.body = nn.Sequential(
            nn.Linear(27, 200),
            nn.ReLU(),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200),
            ResidualBlock(200)
            )
    
        self.dist_head = nn.Sequential(
            nn.Linear(200, 9),
            nn.Softmax(dim=1)
            )
        
        self.eval_head = nn.Sequential(
            nn.Linear(200, 1),
            nn.Tanh()
            )
        
    def forward(self, x):
        x = self.body(x)
        return self.dist_head(x), self.eval_head(x)


class TowardsAttention(nn.Module):
    
    def __init__(self, dimensionality):
        super(TowardsAttention, self).__init__()
        self.d = dimensionality
        # 9 is the number of tokens
        
        self.input_embedder = nn.Sequential(
            #This is not a ffnn. 
            nn.Linear(27, self.d, bias = False), 
            # It embeds tokens to the space of computation
            nn.Flatten()
            )
        
        self.body = nn.Sequential(
            
            ResidualBlock(self.d*9),
            ResidualBlock(self.d*9),
            ResidualBlock(self.d*9),
            ResidualBlock(self.d*9),
            ResidualBlock(self.d*9),
            ResidualBlock(self.d*9),
            ResidualBlock(self.d*9)
            )
    
        self.dist_head = nn.Sequential(
            nn.Linear(self.d*9, 9),
            nn.Softmax(dim=1)
            )
        
        self.eval_head = nn.Sequential(
            nn.Linear(self.d*9, 1),
            nn.Tanh()
            # The output is given wrt the current player
            )
        
    
    def forward(self, x):
        x = self.input_embedder(x)
        x = self.body(x)
        return self.dist_head(x), self.eval_head(x)


class ResidualBlock(nn.Module):
    # Similar to AlphaGoZero residual blocks
    
    def __init__(self, neurons):
        super().__init__()
        
        self.layer_1 = nn.Linear(neurons, neurons)
        self.batch_norm_1 = nn.BatchNorm1d(neurons)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(neurons, neurons)
        self.batch_norm_2 = nn.BatchNorm1d(neurons)
        self.relu_2 = nn.ReLU()
        
    
    def forward(self, x):
        residual = x
        x = self.layer_1(x)
        s = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.batch_norm_2(x)
        
        x = x + residual
        x = self.relu_2(x)
        return x
        
