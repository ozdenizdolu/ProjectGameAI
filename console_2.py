#TODO: More randomness is needed in mcts for nn searchs. Training data
# converges to some types of play.

#TODO: Use convolution, alphazero uses convolution and residual connections.

#BUG-like: It plays the same game over and over on low temperatures

#NOTE: Cross entropy seems to be negative because I show the difference
# from the best possible.

#BUG: Translator is broken! The network does not receive whose turn
# it is. I realized this when I saw that move error decreased but evalaution
# error did not. #BUG !! We want the evaluation wrt white it's why it couldn't
# learn the data. We should either want the evalaution wrt the player
# with the turn, and rotate the board accordingly, or we should 
# clearly make the players accessible to the network. #FIXED
# But why it didn't learn anything? Even though it is unclear whose move
# is that, it still knows the white and black pieces. Even more interestingly
# it learned the move distribution without knowing whose move it is.
# Even if we supplied it lacking information, it should have memorized
# the data! It did not even memorize it!

# Potential #BUG: Does it take the winner moves or all moves for training?
# Or do we include only a fraction of the game states. We should not give
# correlated training examples because it alters the settings SGD is used
# in.


# Experiment 
# 5000 states did not fit even with many epochs. Especially the
# evaluation loss suffers. Body of the network is 7 resnets with
# 300 neurons each. Settings were
# common_settings = {

# 'game': Reversi,
# 'trdev': 'cpu',#training device
# 'mpdev': 'cpu',#mtcs device

# 'loop_count': 5,

# 'd_loss': cross_entropy,
# 'e_loss': mse_loss,

# 'weight_decay': None,

# # Testing Settings

# 'in_mcts_testing': True,
# # MCTS
# 'mcts_step_testing': 10,
# 'move_selector_testing': azms(1),
# 'temperature_testing': 0.05,
# # Evaluator
# 'move_dist_weight_testing': 1.,
# 'evaluation_weight_testing': 0.,

# 'dirichlet_noise_parameter': 0.3,
# 'noise_contribution': 0.25

# }

# varied_settings = {

# # Self-Play Settings
# 'mcts_step_self_play': [256],
# 'move_selector_self_play': [azms(i) for i in [1]],
# 'temperature_self_play': [0.5],

# # Training Settings
# 'max_data_length': [50000],
# 'batches': [2500],
# 'bs': [32],
# 'new_states': [5000],

# 'lr': [0.05],
# 'momentum': [0.9],

# }
#%%

import pprint
from time import sleep
import itertools
import math
import random
import importlib
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import mse_loss

from game.tictactoe import TicTacToe as TCT
from game.reversi import Reversi
from mcts import mcts, uct
from mcts.move_selectors import alpha_zero_move_selector_factory as azms
from mcts.move_selectors import UCT_move_selector_factory
from evaluator.evaluators import random_playout_evaluator
from data_generator import generate_data
from network_translator import (StandardTicTacToeTranslator,
                                StandardReversiTranslator)
# from network_translator import (StandardTicTacToeTranslator,
#                                 StandardReversiTranslator)
from neural_network import ResidualBlock
from game_session import ConsoleGameSession
from agent.agents import (MCTSAgent, user_for_game,
                          RandomAgent, UCTAgent, EvaluatorAgent)
from training_tools import compare
from miscellaneous import cross_entropy


def load_all_uct_data_TCT():
    
    if 'uct_data' in globals():
        raise RuntimeError('It already exists.')
    
    with open('tct_all_UCT_data.pickle','rb') as file:
        global uct_data
        uct_data = pickle.load(file)
        random.shuffle(uct_data)


def discretize_evaluation_TCT(evaluation):
    
    x_bets = evaluation[TCT.X]
    
    if x_bets > 0.5:
        pred_result = 1
    elif x_bets < -0.5:
        pred_result = -1
    else:
        pred_result = 0
    
    return pred_result


def default_network_for_game(game, device):
    # Networks should have their own translators.
    if game == TCT:
        return TCTNetwork(device)
    elif game == Reversi:
        return ReversiNetwork(device)
    else:
        raise NotImplementedError()

def get_evaluator(network, device):
    def evaluator(state, legal_moves = None, player = None):
        translator = network.translator
        
        if legal_moves == None:
            legal_moves = state.moves()
        if player == None:
            player = state.turn()
            
        x = translator.states_to_tensor([state], device)
        move_dist, evaluation = network(x)
        return (
            translator.tensor_to_dists([state], [legal_moves], move_dist)[0], 
            translator.tensor_to_evals([state], evaluation)[0])
    return evaluator



class ReversiNetwork(nn.Module):
    
    def __init__(self, device):
        super(ReversiNetwork, self).__init__()
        
        self.translator = StandardReversiTranslator()
        
        self.body = nn.Sequential(
            nn.Linear(3*8*8, 1000),
            nn.ReLU(),
            ResidualBlock(1000, device),
            ResidualBlock(1000, device),
            ResidualBlock(1000, device),
            )
    
        self.dist_head = nn.Sequential(
            nn.Linear(1000, 8*8 + 1, device=device),
            nn.Softmax(dim=1),
            )
        
        self.eval_head = nn.Sequential(
            nn.Linear(1000, 1, device=device),
            nn.Tanh(),
            )
        
    def forward(self, x):
        x = self.body(x)
        return self.dist_head(x), self.eval_head(x)

class Training_Environment:
    
    def __init__(self,
                game,
                net,
                loop_count,
                mpdev,
                trdev,
                d_loss,
                e_loss,
                mcts_step_self_play,
                move_selector_self_play,
                temperature_self_play,
                lr,
                weight_decay,
                momentum,
                bs,
                in_mcts_testing,
                mcts_step_testing,
                move_selector_testing,
                temperature_testing,
                move_dist_weight_testing,
                evaluation_weight_testing,
                max_data_length,
                batches,
                new_states,
                dirichlet_noise_parameter,
                noise_contribution
                ):
        
        if weight_decay != None:
            raise NotImplementedError('''Weight decay currently applies to all
                                      parameters, not just weights. Update the
                                      code to use this functionality securely.''')
        else:
            weight_decay = 0.
        
        self.game = game
        self.net = net
        self.loop_count = loop_count
        self.mpdev = mpdev
        self.trdev = trdev
        self.d_loss = d_loss
        self.e_loss = e_loss
        self.mcts_step_self_play = mcts_step_self_play
        self.move_selector_self_play = move_selector_self_play
        self.temperature_self_play = temperature_self_play
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.bs = bs
        self.in_mcts_testing = in_mcts_testing
        self.mcts_step_testing = mcts_step_testing
        self.move_selector_testing = move_selector_testing
        self.temperature_testing = temperature_testing
        self.move_dist_weight_testing = move_dist_weight_testing
        self.evaluation_weight_testing = evaluation_weight_testing
        self.max_data_length = max_data_length
        self.batches = batches
        self.new_states = new_states
        self.dirichlet_noise_parameter = dirichlet_noise_parameter
        self.noise_contribution = noise_contribution
        
        
        self.random_win_rates = []
        self.evaluator_random_win_rates =[]
        self.pool_dist_loss_rates = []
        self.pool_eval_loss_rates = []
        self.uct_100_win_rates = []
        self.uct_1000_win_rates = []
        self.data_test = []
        self.pool_length_record = []
        
        self.test_in()
        
    def test_in(self):
        
        print("Testing is starting... ")
        self.net.eval()
        "#TODO should you put the device to mpdev here?"
        self.net.to(self.mpdev)
        
        raw_ev = get_evaluator(self.net, self.mpdev)
        
        evaluator_agent = EvaluatorAgent(raw_ev,
                   distribution_weight=self.move_dist_weight_testing,
                   evaluation_weight=self.evaluation_weight_testing)
        
        mcts_agent = MCTSAgent(raw_ev,
                          self.mcts_step_testing,
                          self.move_selector_testing,
                          self.temperature_testing)
        
        if self.in_mcts_testing:
            default_agent = mcts_agent
        else:
            default_agent = evaluator_agent
        
        self.random_win_rates.append(
            compare(self.game,
                    [default_agent, RandomAgent()], 20)[default_agent]/20.)
        
        self.evaluator_random_win_rates.append(
            compare(self.game,
                    [evaluator_agent, RandomAgent()], 20)[evaluator_agent]/20.)
       
        #keep track of the loss per example in the pool
        if hasattr(self, 'x_pool'):
            with torch.no_grad():
                optimum_dist_loss = self.d_loss(
                                        self.dist_pool, self.dist_pool)
                
                net_dist, net_eval = self.net(self.x_pool)
                
                dist_loss = (self.d_loss(net_dist, self.dist_pool)
                             - optimum_dist_loss)
                eval_loss = self.e_loss(net_eval, self.eval_pool)
                
                pool_dist_loss = dist_loss.item()
                pool_eval_loss = eval_loss.item()
                pool_length = self.x_pool.shape[0]/self.max_data_length
        else:
            pool_dist_loss = -1
            pool_eval_loss = -1
            pool_length = -1
       
        self.pool_dist_loss_rates.append(pool_dist_loss)
        self.pool_eval_loss_rates.append(pool_eval_loss)
        self.pool_length_record.append(pool_length)
       
        # We did not support mcts giving game evaluations
        # self.data_test_with_mcts.append(
        #     sum(1 if (discretize_evaluation(evaluation) == 
        #               discretize_evaluation(
        #                   mcts(state, raw_ev, self.mcts_step_testing,
        #                        self.move_selector_testing,
        #                        self.temperature_testing,
        #                        ,return_type)) else 0
        #     for state, move, evaluation in current_data)/sample_size)
        # )
        
        fig, ax = plt.subplots()
        
        for f,word, label in zip(
            [self.random_win_rates,
             self.pool_dist_loss_rates,
             self.pool_eval_loss_rates,
             self.pool_length_record,
             self.evaluator_random_win_rates,
             # self.uct_100_win_rates,
             # self.uct_1000_win_rates,
             # self.data_test
             ],
            ['green','orange','red', 'purple', 'blue', 'black'],
            ['random', 'dist_loss', 'eval_loss', 'data_fulness',
             'eval_vs_random']):
            ax.plot(range(len(f)), f, 'tab:{}'.format(word), label=label)
        plt.legend(loc="upper left")
        plt.show()
    
    def training(self):
        
        for i in range(1, 1 + self.loop_count):
            print("\nLoop number {}/{} is starting...\n".format(i, self.loop_count))
            
            # Add to the pool via self-play
            self.net.eval()
            self.net.to(device=self.mpdev)
            raw_data = generate_data(
                self.game,
                's',
                self.new_states,
                get_evaluator(self.net, self.mpdev),
                self.mcts_step_self_play,
                self.move_selector_self_play,
                self.temperature_self_play,
                dirichlet_noise_parameter=self.dirichlet_noise_parameter,
                noise_contribution=self.noise_contribution,
                shuffle=True)
            
            x, dist, eva = self.net.translator.translate_data(
                raw_data, self.trdev)
            
            #Initialize here because shape is unknown before translation
            if hasattr(self, 'x_pool'):
                # add the data and clip if there is excess
                
                excess_data_length = (x.size(dim=0)
                                      + self.x_pool.size(dim = 0)
                                      - self.max_data_length)
                
                if excess_data_length > 0:
                    self.x_pool = self.x_pool[excess_data_length:]
                    self.dist_pool = self.dist_pool[excess_data_length:]
                    self.eval_pool = self.eval_pool[excess_data_length:]
                
                self.x_pool = torch.cat((self.x_pool, x), dim = 0)
                self.dist_pool = torch.cat((self.dist_pool, dist), dim = 0)
                self.eval_pool = torch.cat((self.eval_pool, eva), dim = 0)
            
            else:
                if x.shape[0] > self.max_data_length:
                    raise ValueError('''max data length is smaller than
                                     the possible amount that can be received.
                                     ''')
                self.x_pool = x
                self.dist_pool = dist
                self.eval_pool = eva
            
            # Backprop training
            self.net.train()
            self.net.to(device=self.trdev)
            
            # Use the pool to train the agent.
            
            # Preperation before batches
            
            if not (self.x_pool.device == self.dist_pool.device
                    == self.eval_pool.device):
                raise ValueError('Device problem.')
            
            optimizer = torch.optim.SGD(self.net.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
            
            dist_loss_tracker = []
            eval_loss_tracker = []
            dist_optimum_tracker = []
            # Batch loop
            
            optimum_dist_loss = self.d_loss(
                self.dist_pool, self.dist_pool)
            
            for j in range(self.batches):                
                #TODO: make sure this is not a bottleneck when using GPU
                batch_indices = torch.tensor(
                    random.sample(range(self.x_pool.shape[0]), self.bs),
                    device=self.trdev)
                batch = [torch.index_select(tensor, 0, batch_indices)
                          for tensor in 
                          (self.x_pool, self.dist_pool, self.eval_pool)]
                
                batch_x, batch_dist, batch_eval = batch
                
                net_dist, net_eval = self.net(batch_x)
                
                dist_loss = (self.d_loss(net_dist, batch_dist)
                             - optimum_dist_loss)
                eval_loss = self.e_loss(net_eval, batch_eval)
                
                loss = dist_loss + eval_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                dist_loss_tracker.append(dist_loss.item())
                eval_loss_tracker.append(eval_loss.item())
            
            # Graph the loss
            
            fig, ax = plt.subplots()
            
            for f,word in zip(
                [dist_loss_tracker,
                 eval_loss_tracker,
                 ],
                ['green','orange','red', 'purple', 'blue', 'black']):
                ax.plot(range(len(f)), f, 'tab:{}'.format(word))
            plt.show()
            
            # Test
            self.test_in()
        # Loop ended
        return
    
    # TODO: Add more methods which support the development, and let
    # us do the training parts seperately

    def generate_games(self, num_of_states, raw=False):
        
        self.net.eval()
        self.net.to(device=self.mpdev)
        raw_data = generate_data(
            self.game,
            's',
            num_of_states,
            get_evaluator(self.net, self.mpdev),
            self.mcts_step_self_play,
            self.move_selector_self_play,
            self.temperature_self_play,
            dirichlet_noise_parameter=self.dirichlet_noise_parameter,
            noise_contribution=self.noise_contribution,
            shuffle=True)
        
        if raw:
            return raw_data
        
        x, dist, eva = self.net.translator.translate_data(
            raw_data, self.trdev)
        
        return (x, dist, eva)
    
    def train_data(self, graph=True):
        train_data(env.net, env.x_pool, env.dist_pool, env.eval_pool,
                   env.lr, env.momentum, env.bs, env.batches, env.trdev,
                   env.d_loss, env.e_loss, graph)

#TODO WHY 3 train methods?
def train_data(net,
               data_x,
               data_dist,
               data_eval,
               lr,
               momentum,
               bs,
               batches,
               device,
               d_loss,
               e_loss,
               graph = True,
               record_frequency = -1):
    """
    

    Parameters
    ----------
    record_frequency : TYPE, optional
        If -1 or not given then the default is 1 per 100

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if record_frequency == -1:
        record_frequency = 100
    
    def record():
        # compute all data and find the training loss
        with torch.no_grad():
            net_dist, net_eval = net(data_x)
            
            dist_loss = (d_loss(net_dist, data_dist) 
                         - optimum_dist_loss)
            eval_loss = e_loss(net_eval, data_eval)
        
            dist_loss_tracker.append(dist_loss.item())
            eval_loss_tracker.append(eval_loss.item())
    
    net.train()
    net.to(device=device)
    
    if not (data_x.device == data_dist.device == data_eval.device):
        raise ValueError('Device mismatch.')
        
    if not (data_x.shape[0] == data_dist.shape[0] == data_eval.shape[0]):
        raise ValueError('Shape mismatch.')
    
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr,
                                momentum=momentum)
    
    dist_loss_tracker = []
    eval_loss_tracker = []
    # Batch loop
    
    optimum_dist_loss = d_loss(data_dist, data_dist)
    
    # Record at initialization.
    record()

    for j, record_counter in zip(range(batches), itertools.count(1)):      
        #TODO: make sure this is not a bottleneck when using GPU
        batch_indices = torch.tensor(
            random.sample(range(data_x.shape[0]), bs),
            device=device)
        batch = [torch.index_select(tensor, 0, batch_indices)
                  for tensor in 
                  (data_x,
                  data_dist,
                  data_eval)]
        
        batch_x, batch_dist, batch_eval = batch
        
        net_dist, net_eval = net(batch_x)
        
        dist_loss = (d_loss(net_dist, batch_dist) 
                     - optimum_dist_loss)
        eval_loss = e_loss(net_eval, batch_eval)
        
        loss = dist_loss + eval_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record at given frequency and at the end
        if (record_counter % record_frequency == 0) or (j == batches - 1):
            record()
            if graph:
                fig, ax = plt.subplots()
                
                for f,word in zip(
                    [dist_loss_tracker,
                     eval_loss_tracker,
                     ],
                    ['green','orange','red', 'purple', 'blue', 'black']):
                    ax.plot(range(len(f)), f, 'tab:{}'.format(word))
                plt.show()
    
    
    


#%%

        
load_all_uct_data_TCT()

# Settings

common_settings = {

'game': Reversi,
'trdev': 'cpu',#training device
'mpdev': 'cpu',#mtcs device

'loop_count': 10,

'd_loss': cross_entropy,
'e_loss': mse_loss,

'weight_decay': None,

# Testing Settings

'in_mcts_testing': True,
# MCTS
'mcts_step_testing': 10,
'move_selector_testing': azms(1),
'temperature_testing': 0.05,
# Evaluator
'move_dist_weight_testing': 1.,
'evaluation_weight_testing': 0.,

'dirichlet_noise_parameter': 0.3,
'noise_contribution': 0.25

}

varied_settings = {

# Self-Play Settings
'mcts_step_self_play': [128],
'move_selector_self_play': [azms(i) for i in [1]],
'temperature_self_play': [1],

# Training Settings
'max_data_length': [50000],
'batches': [5000],
'bs': [64],
'new_states': [5000],

'lr': [0.001],
'momentum': [0.9],

}

if 'env' in globals():
    print('Environment already exists! Did not change a thing')
else:
    env = Training_Environment(**{  'net': default_network_for_game(
                    common_settings['game'],
                    common_settings['mpdev']),
                **common_settings,
                **{setting: random.choice(values)
                   for setting, values in varied_settings.items()}})
    
#%%

#Console shortcuts

ev = get_evaluator(env.net, env.mpdev)
mcts_ag = MCTSAgent(ev, env.mcts_step_testing,
                  env.move_selector_testing, env.temperature_testing)
dist_ag = EvaluatorAgent(ev)
monte = lambda x: mcts(x, ev, env.mcts_step_testing,env.move_selector_testing
                       ,env.temperature_testing, return_type = 'distribution')
user = user_for_game(env.game)


#%%

pprint.pprint(env.__dict__)

# env.training()

