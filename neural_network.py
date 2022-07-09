"""
Development note:
    
Refactoring will be needed at some point.
current_device parameter seems to be obsolete.
divide translator from the nn
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class TicTacToe_defaultNN(nn.Module):
    
    def __init__(self, device):
        super(TicTacToe_defaultNN, self).__init__()
        
        self.body = nn.Sequential(
            nn.Linear(27, 200, device=device),
            nn.ReLU(),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device)
            )
    
        self.dist_head = nn.Sequential(
            nn.Linear(200, 9, device=device),
            nn.Softmax(dim=1)
            )
        
        self.eval_head = nn.Sequential(
            nn.Linear(200, 1, device=device),
            nn.Tanh()
            )
        
    def forward(self, x):
        x = self.body(x)
        return self.dist_head(x), self.eval_head(x)


class TowardsAttention(nn.Module):
    
    def __init__(self, dimensionality, device):
        super(TowardsAttention, self).__init__()
        self.d = dimensionality
        # 9 is the number of tokens
        
        self.input_embedder = nn.Sequential(
            #This is not a ffnn. 
            nn.Linear(27, self.d, bias = False, device=device), 
            # It embeds tokens to the space of computation
            nn.Flatten()
            )
        
        self.body = nn.Sequential(
            
            ResidualBlock(self.d*9, device=device),
            ResidualBlock(self.d*9, device=device),
            ResidualBlock(self.d*9, device=device),
            ResidualBlock(self.d*9, device=device),
            ResidualBlock(self.d*9, device=device),
            ResidualBlock(self.d*9, device=device),
            ResidualBlock(self.d*9, device=device)
            )
    
        self.dist_head = nn.Sequential(
            nn.Linear(self.d*9, 9, device=device),
            nn.Softmax(dim=1)
            )
        
        self.eval_head = nn.Sequential(
            nn.Linear(self.d*9, 1, device=device),
            nn.Tanh()
            # The output is given wrt the current player
            )
        
    
    def forward(self, x):
        x = self.input_embedder(x)
        x = self.body(x)
        return self.dist_head(x), self.eval_head(x)

class NetworkWithAttention(nn.Module):
    #Assumes one-hot coded 27 dimensional input. bs x 9 x 27, in fact.
    
    def __init__(self, device):
        super().__init__()
        # 9 is the number of tokens
        
        self.input_embedder = nn.Linear(27, 20,
                                        bias = False, device=device)
        
        self.body = nn.Sequential(
            
            WithResidual(
                SelfAttentionBlock(20, 20, 20,device=device)),
            
            nn.Flatten(),
            
            nn.BatchNorm1d([180], device=device),
            
            nn.Unflatten(-1, (9,20)),
            
            WithResidual(
                nn.Sequential(
                    nn.Linear(20, 100, device=device),
                    nn.ReLU(),
                    nn.Linear(100, 20, device=device),
                    nn.ReLU()
                    )),
            
            nn.Flatten(),
            
            nn.BatchNorm1d([180], device=device),
            
            nn.Unflatten(-1, (9,20)),
            
            WithResidual(
                SelfAttentionBlock(20, 20, 20,device=device)),
            
            nn.Flatten(),
            
            nn.BatchNorm1d([180], device=device),
            
            nn.Unflatten(-1, (9,20)),
            
            WithResidual(
                nn.Sequential(
                    nn.Linear(20, 100, device=device),
                    nn.ReLU(),
                    nn.Linear(100, 20, device=device),
                    nn.ReLU()
                    )),
            
            nn.Flatten(),
            
            nn.BatchNorm1d([180], device=device),
            
            nn.Unflatten(-1, (9,20)),
            
            WithResidual(
                SelfAttentionBlock(20, 20, 20,device=device)),
            
            nn.Flatten(),
            
            nn.BatchNorm1d([180], device=device),
            
            nn.Unflatten(-1, (9,20)),
            
            WithResidual(
                nn.Sequential(
                    nn.Linear(20, 100, device=device),
                    nn.ReLU(),
                    nn.Linear(100, 20, device=device),
                    nn.ReLU()
                    )),
            
           nn.Flatten(),
           
           nn.BatchNorm1d([180], device=device),
           
           nn.Unflatten(-1, (9,20)),
            
            WithResidual(
                SelfAttentionBlock(20, 20, 20,device=device)),
            
            nn.Flatten(),
            
            nn.BatchNorm1d([180], device=device),
            
            nn.Unflatten(-1, (9,20)),
            
            WithResidual(
                nn.Sequential(
                    nn.Linear(20, 100, device=device),
                    nn.ReLU(),
                    nn.Linear(100, 20, device=device),
                    nn.ReLU()
                    )),
            
            nn.Flatten(),
            
            nn.BatchNorm1d([180], device=device),
            
            nn.Linear(20*9, 100, device = device),
            
            nn.ReLU()
            
            )
    
        self.dist_head = nn.Sequential(
            nn.Linear(100, 9, device=device),
            nn.Softmax(dim=1)
            )
        
        self.eval_head = nn.Sequential(
            nn.Linear(100, 1, device=device),
            nn.Tanh()
            # The output is given wrt the current player
            )
    
    def forward(self, x):
        x = self.input_embedder(x)
        x = self.body(x)
        return self.dist_head(x), self.eval_head(x)
        

class WithResidual(nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        output = self.module(x)
        if output.shape != x.shape:
            raise RuntimeError('''Residual connection cannot be made
                               if the output shape do not match the
                               input shape.''')
        return torch.add(output, x)

#In progress
# class MultiHeadSelfAttentionBlock(nn.Module):
    
#     def __init__(self, token_dimensionality, num_of_heads):
#         super().__init__()
        
#         self.dim = token_dimensionality
#         self.h = num_of_heads
        
#         if (self.dim % self.h) != 0:
#             raise ValueError()
        
#         self.hidden_dimension = self.dim // self.h
        
#         self.attention_heads = 
            
    
#     def forward(self, x):
#         # x is of form (N, T, dim)
#         if len(x.shape) != 3 or x.shape[-1] != self.dim:
#             raise ValueError()
        
        
        

class SelfAttentionBlock(nn.Module):
    
    def __init__(self, input_token_dimension, query_dimension, output_token_dimension, device):
        super().__init__()
        
        self.input_token_dimension = input_token_dimension
        self.query_dimension = query_dimension
        self.output_token_dimension = output_token_dimension
        
        self.query_network = nn.Linear(self.input_token_dimension,
                                       self.query_dimension,
                                       device=device)
        
        self.key_network = nn.Linear(self.input_token_dimension,
                                     self.query_dimension,
                                     device=device)
        
        self.value_network = nn.Linear(self.input_token_dimension,
                                       self.output_token_dimension,
                                       device=device)
        
        
    def forward(self, x):
        # 0th dimension is batch size.
        # Next dimension is token ordering.
        # Next dimension is the information of the token.
        
        # self has Query, Key, and Value networks which calculate
        # the respective trait of the token.
        
        Q = self.query_network(x)
        K = self.key_network(x)
        V = self.value_network(x)
        
        pre_processed_attention = torch.matmul(Q, torch.transpose(K, 1, 2))
        attention = torch.nn.functional.softmax(
            torch.div(pre_processed_attention,
                      math.sqrt(self.query_dimension)),
            dim=-1)
        
        # attention matmul V gives 3 dimensional array
        # First dimension is batch size
        # Second one is token
        # Third contains the result of the token
        return torch.matmul(attention, V)
        

class ResidualBlock(nn.Module):
    # Similar to AlphaGoZero residual blocks
    
    def __init__(self, neurons, device):
        super().__init__()
        
        self.layer_1 = nn.Linear(neurons, neurons, device=device)
        self.batch_norm_1 = nn.BatchNorm1d(neurons, device=device)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(neurons, neurons, device=device)
        self.batch_norm_2 = nn.BatchNorm1d(neurons, device=device)
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
        
