"""
Development note:
    
Refactoring will be needed at some point.
current_device parameter seems to be obsolete.
divide translator from the nn
"""
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

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
    
class NetworkWithMultiHeadAttention(nn.Module):
    
    def __init__(self, dim, heads, device, intra_expansion = 5):
        super().__init__()
        self.d = dim
        self.h = heads
        
        self.input_embedder = nn.Linear(27, dim,
                                        bias = False, device=device)
        
        self.body = nn.Sequential(*[
            TransformerEncoder(dim, heads, device,
                               intra_expansion=intra_expansion)
            for i in range(6)]
            )
        
        self.dist_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9*dim, 9, device=device),
            nn.Softmax(dim=1)
            )
        
        self.eval_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9*dim, 1, device=device),
            nn.Tanh()
            )
    
    def forward(self, x):
        x = self.input_embedder(x)
        x = self.body(x)
        return self.dist_head(x), self.eval_head(x)
        
class TransformerEncoder(nn.Module):
    
    def __init__(self, dim, heads, device, intra_expansion = 5):
        super().__init__()
        self.d = dim
        self.h = heads
        self.body = nn.Sequential(
        
            WithResidual(
                MultiHeadSelfAttentionBlock(dim, heads, device=device)
                ),
            
            nn.LayerNorm((dim,), device=device),
            
            WithResidual(
                nn.Sequential(
                    nn.Linear(dim, dim*intra_expansion, device=device),
                    nn.ReLU(),
                    nn.Linear(dim*intra_expansion, dim, device=device))
                ),
            
            nn.LayerNorm((dim,), device=device))
        
    
    def forward(self, x):
        return self.body(x)
        
        

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

class MultiHeadSelfAttentionBlock(nn.Module):
    
    def __init__(self, token_dimensionality, num_of_heads, device = None,
                 linear_layer_at_end = True):
        super().__init__()
        
        if device is None:
            raise ValueError('Please specify a device.')
        
        self.dim = token_dimensionality
        self.h = num_of_heads
        
        if (self.dim % self.h) != 0:
            raise ValueError()
        
        self.hidden_dim = self.dim // self.h
        
        self.query_heads = nn.Parameter(torch.empty(
            (self.h, self.hidden_dim, self.dim), device = device))
        
        self.key_heads = nn.Parameter(torch.empty(
            (self.h, self.hidden_dim, self.dim), device = device))
        
        self.value_heads = nn.Parameter(torch.empty(
            (self.h, self.hidden_dim, self.dim), device = device))
        
        if linear_layer_at_end:
            self.end_linear = nn.Linear(self.dim, self.dim, bias = False,
                                        device=device)
        else:
            self.register_parameter('end_linear', None)
            
        self.initialize_parameters()
        
    def initialize_parameters(self):
        assert len(list(self.parameters())) == 4
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter)
    
    def forward(self, x):
        # x is of form (N, T, dim)
        if len(x.shape) != 3 or x.shape[-1] != self.dim:
            raise ValueError()
            
        N = x.shape[0]
        T = x.shape[1]
        
        x = x.unsqueeze(1) # now (N, 1, T, dim)
        x = x.unsqueeze(-1) # now (N, 1, T, dim, 1)

        query_matrix = self._broadcast_head(self.query_heads)
        key_matrix = self._broadcast_head(self.key_heads)
        value_matrix = self._broadcast_head(self.value_heads)
        
        queries = torch.einsum('abcij,abcjk->abci', query_matrix, x)
        keys = torch.einsum('abcij,abcjk->abci', key_matrix, x)
        values = torch.einsum('abcij,abcjk->abci', value_matrix, x)


        assert (tuple(queries.shape)
                == tuple(keys.shape)
                == tuple(values.shape)
                == (N, self.h, T, self.hidden_dim))
        
        attention = F.softmax(
            torch.div(
            torch.matmul(queries, torch.transpose(keys,-1,-2)),
            # torch.einsum('abij,abjk->abik', queries, torch.transpose(keys,-1,-2)),
            math.sqrt(self.hidden_dim)), dim = -1)
        
        # The product is of shape (N, H, T, d_k)
        head_outputs = torch.matmul(attention, values)
        
        concatenated = torch.flatten(
            torch.transpose(head_outputs, 1, 2),
            -2, -1)
        
        if self.end_linear is not None:
            return self.end_linear(concatenated)
        else:
            return concatenated
        

    def _broadcast_head(self, head):
        return head.unsqueeze(1).unsqueeze(0)

        

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
        
