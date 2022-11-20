#TODO: More randomness is needed in mcts for nn searchs. Training data
# converges to some types of play.

#TODO: Put Dirichlet noise on the root moves like AlphaZero

#TODO: keep a pool of games for the training.

#TODO: Use cross entropy loss for move distributions

#BUG: It plays the same game over and over on low temperatures
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
from network_translator import StandardTicTacToeTranslator
# from network_translator import (StandardTicTacToeTranslator,
#                                 StandardReversiTranslator)
from neural_network import ResidualBlock
from game_session import ConsoleGameSession
from agent.agents import (MCTSAgent, user_for_game,
                          RandomAgent, UCTAgent, EvaluatorAgent)
from training_tools import compare
from miscellaneous import cross_entropy


def load_all_uct_data():
    
    if 'uct_data' in globals():
        raise RuntimeError('It already exists.')
    
    with open('tct_all_UCT_data.pickle','rb') as file:
        global uct_data
        uct_data = pickle.load(file)
        random.shuffle(uct_data)
        
        # training_data = raw_data[0:15000]
        # test_data = raw_data[15000:18000]
        # validation_data = raw_data[18000:]
        
        # trd = tr.translate_data(training_data, device)
        # ted = tr.translate_data(test_data, device)
        # vad = tr.translate_data(validation_data, device)


def discretize_evaluation(evaluation):
    
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

#TODO
# class ReversiNetwork(nn.Module):
    
#     def __init__(self, device):
#         super(ReversiNetwork, self).__init__()
        
#         self.translator = StandardReversiTranslator()
        
#         self.body = nn.Sequential(
#             nn.Linear(#TODO, 200, device=device),
#             nn.ReLU(),
#             ResidualBlock(200, device=device),
#             ResidualBlock(200, device=device)
#             )
            
#         self.dist_head = nn.Sequential(
#             nn.Linear(200, #TODO, device=device),
#             nn.Softmax(dim=1)
#             )
        
#         self.eval_head = nn.Sequential(
#             nn.Linear(200, #TODO, device=device),
#             nn.Tanh()
#             )

#     def forward(self, x):
#         x = self.body(x)
#         return self.dist_head(x), self.eval_head(x)

class TCTNetwork(nn.Module):
    
    def __init__(self, device):
        super(TCTNetwork, self).__init__()
        
        self.translator = StandardTicTacToeTranslator()
        
        self.body = nn.Sequential(
            nn.Linear(27, 200, device=device),
            nn.ReLU(),
            ResidualBlock(200, device=device),
            ResidualBlock(200, device=device),
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


# class TCTNNEvaluator():
#     """This is an evaluator which works by using a raw neural network."""
    
#     def __init__(self, nn, device):
#         self._net = nn
#         self.device = device
    
#     def __call__(self, game_state, legal_moves, turn):
        
#         translator = self._net.translator
        
#         states = [game_state]
#         in_ = translator.states_to_tensor(states, self.device)
#         dist_, eval_ = self._net(in_)
        
#         move_dist = translator.tensor_to_dists(states, [legal_moves], dist_)[0]
#         evaluation = translator.tensor_to_evals(states, eval_)[0]
        
#         return move_dist, evaluation 
    

def train_data(net, epochs, data,
                      move_loss_function,
                      eval_loss_function,
                      lr=0.01, momentum = 0.9, weight_decay=None,
                      batch_size=32):
        
    if weight_decay != None:
        raise NotImplementedError('''Weight decay currently applies to all
                                  parameters, not just weights. Update the
                                  code to use this functionality securely.''')
    else:
        weight_decay = 0.
    
    x_data, dist_data, eval_data = data
    if not (x_data.device == dist_data.device == eval_data.device):
        raise ValueError('Device problem.')
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    
    num_of_batches_per_epoch = math.ceil(len(x_data)/batch_size)
    
    # start training
    net.eval()
    print('Training is starting...\n')
    
    net.train()
    
    for epoch in range(1, epochs+1):
        for _ in range(num_of_batches_per_epoch):
            #TODO: make sure this is not a bottleneck when using GPU
            example_indices = torch.tensor(random.sample(
                range(x_data.shape[0]), batch_size), device=x_data.device)
            batch = [torch.index_select(tensor, 0, example_indices)
                      for tensor in (x_data, dist_data, eval_data)]
            
            x, dist_target, eval_target = batch
            
            net_dist, net_eval = net(x)
            
            dist_loss = move_loss_function(net_dist, dist_target)
            eval_loss = eval_loss_function(net_eval, eval_target)
            
            loss = dist_loss + eval_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        net.eval()
        print('Epoch {}/{} complete.'.format(epoch,epochs))
        
        net.train()
        
    net.eval()


# def training(game,
#              net,
#              loop_count,
#              mcts_play_device,
#              training_device,
#              distribution_loss,
#              evaluation_loss,
#              mcts_step_self_play,
#              move_selector_self_play,
#              temperature_self_play,
#              training_data_goal,
#              epochs_per_loop,
#              lr,
#              weight_decay,
#              momentum,
#              bs,
#              in_mcts_testing,
#              mcts_step_testing,
#              move_selector_testing,
#              temperature_testing,
#              move_dist_weight_testing,
#              evaluation_weight_testing
#              ):
    
#     # Shortcuts
#     mpdev = mcts_play_device
#     trdev = training_device
    
#     # Records
    
#     random_win_rates = []
#     uct_100_win_rates = []
#     uct_1000_win_rates = []
    
#     def test_in():
#         # Tests
        
#         print("Testing is starting... ")
#         net.eval()
    
#         if in_mcts_testing:
#             agent = MCTSAgent(get_evaluator(net, mpdev),
#                               mcts_step_testing,
#                               move_selector_testing,
#                               temperature_testing)
#         else:
#             agent = EvaluatorAgent(get_evaluator(net, mpdev),
#                        distribution_weight=move_dist_weight_testing,
#                        evaluation_weight=evaluation_weight_testing)
        
#         random_win_rates.append(
#             compare(game, [agent, RandomAgent()], 2)[agent]/2.)
        
#         uct_100_win_rates.append(
#             compare(game, [agent, UCTAgent(100)], 2)[agent]/2.)
        
#         uct_1000_win_rates.append(
#             compare(game, [agent, UCTAgent(1000)], 2)[agent]/2.)
        
#         fig, ax = plt.subplots()
        
        
#         for f,word in zip(
#             [random_win_rates, uct_100_win_rates, uct_1000_win_rates],
#             ['green','orange','red']):
#             ax.plot(range(len(f)), f, 'tab:{}'.format(word))
#         plt.show()
    
    
#     # Test First
#     test_in()
    
#     for i in range(1, 1 + loop_count):
#         print("\nLoop number {}/{} is starting...\n".format(i, loop_count))
        
#         # Self-Play
#         net.eval()
#         net.to(device=mpdev)
#         raw_data = generate_data(game,
#                                  's',
#                                  training_data_goal,
#                                  get_evaluator(net, mpdev),
#                                  mcts_step_self_play,
#                                  move_selector_self_play,
#                                  temperature_self_play)
        
#         # Backprop training
#         net.train()
#         data = net.translator.translate_data(raw_data, trdev)
#         net.to(device=trdev)
#         train_data(
#             net,
#             epochs_per_loop,
#             data,
#             distribution_loss,
#             evaluation_loss,
#             lr=lr,
#             momentum=momentum,
#             weight_decay=weight_decay,
#             batch_size=bs)
        
#         # Test
#         test_in()


class Training_Environment:
    
    def __init__(self,
                game,
                net,
                loop_count,
                mpdev,
                trdev,
                distribution_loss,
                evaluation_loss,
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
        self.distribution_loss = distribution_loss
        self.evaluation_loss = evaluation_loss
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
        self.uct_100_win_rates = []
        self.uct_1000_win_rates = []
        self.data_test = []
        
        self.test_in()
        
    def test_in(self):
        
        print("Testing is starting... ")
        self.net.eval()
        
        raw_ev = get_evaluator(self.net, self.mpdev)
        
        if self.in_mcts_testing:
            agent = MCTSAgent(raw_ev,
                              self.mcts_step_testing,
                              self.move_selector_testing,
                              self.temperature_testing)
        else:
            agent = EvaluatorAgent(raw_ev,
                       distribution_weight=self.move_dist_weight_testing,
                       evaluation_weight=self.evaluation_weight_testing)
        
        self.random_win_rates.append(
            compare(self.game, [agent, RandomAgent()], 10)[agent]/10.)
        
        self.uct_100_win_rates.append(
            compare(self.game, [agent, UCTAgent(100)], 10)[agent]/10.)
        
        self.uct_1000_win_rates.append(
            compare(self.game, [agent, UCTAgent(1000)], 1)[agent]/1.)
        
        # Performance on data
        
        #pick random 500 training data
        sample_size = 500
        current_data = random.sample(uct_data, sample_size)
        
        self.data_test.append(
            sum(1 if (discretize_evaluation(evaluation) == 
                      discretize_evaluation(get_evaluator(
                          self.net, self.mpdev)(state)[1])) else 0
            for state, move, evaluation in current_data)/sample_size)
        
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
        
        for f,word in zip(
            [self.random_win_rates,
             self.uct_100_win_rates,
             self.uct_1000_win_rates,
             self.data_test],
            ['green','orange','red', 'purple', 'blue', 'black']):
            ax.plot(range(len(f)), f, 'tab:{}'.format(word))
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
            try:
                self.x_pool
            except AttributeError:
                self.x_pool = x
                self.dist_pool = dist
                self.eval_pool = eva
            
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
            
            
            # Backprop training
            self.net.train()
            self.net.to(device=self.trdev)
            
            # Use the pool to train the agent.
            
            if not (self.x_pool.device == self.dist_pool.device == self.eval_pool.device):
                raise ValueError('Device problem.')
            
            optimizer = torch.optim.SGD(self.net.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
            
            
            for j in range(self.batches):
                #TODO: train with a batch from the pool
                
                #TODO: make sure this is not a bottleneck when using GPU
                batch_indices = torch.tensor(
                    random.sample(range(self.x_pool.shape[0]), self.bs),
                    device=self.trdev)
                batch = [torch.index_select(tensor, 0, batch_indices)
                          for tensor in 
                          (self.x_pool, self.dist_pool, self.eval_pool)]
                
                batch_x, batch_dist, batch_eval = batch
                
                net_dist, net_eval = self.net(batch_x)
                
                dist_loss = self.distribution_loss(net_dist, batch_dist)
                eval_loss = self.evaluation_loss(net_eval, batch_eval)
                
                loss = dist_loss + eval_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Test
            self.test_in()
        # End of Loop
        return
        
    



#%%

        
load_all_uct_data()

# Settings

common_settings = {

'game': TCT,
'trdev': 'cpu',#training device
'mpdev': 'cpu',#mtcs device

'loop_count': 5,

'distribution_loss': cross_entropy,
'evaluation_loss': mse_loss,

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
'mcts_step_self_play': [256],
'move_selector_self_play': [azms(i)
                           for i in [1]],
'temperature_self_play': [1],

# Training Settings
'max_data_length': [2000],
'batches': [10],
'bs': [20],
'new_states': [100],

'lr': [0.005],
'momentum': [0.9],

}

try:
    env
    print('Environment already exists! Did not change a thing')
except NameError:
    env = Training_Environment(**{  'net': default_network_for_game(
                    common_settings['game'],
                    common_settings['mpdev']),
                **common_settings,
                **{setting: random.choice(values)
                   for setting, values in varied_settings.items()}})
    
#%%

ev = get_evaluator(env.net, env.mpdev)
ag = MCTSAgent(ev, env.mcts_step_testing,
                  env.move_selector_testing, env.temperature_testing)
monte = lambda x: mcts(x, ev, env.mcts_step_testing,env.move_selector_testing
                       ,env.temperature_testing, return_type = 'distribution')
user = user_for_game(env.game)


#%%

pprint.pprint(env.__dict__)


# OLD 

#%%
# # General Settings
# game = TCT
# training_device = trdev = 'cpu'
# mcts_play_device = mpdev = 'cpu'


#%%

# # Initialization
# try:
#     initialized
# except NameError:
    
#     net = default_network_for_game(game, mpdev)
#     net.eval() # Set to evaluation by default
#     # ev = get_evaluator(net, device)
#     # ag = MCTSAgent(ev, 50, azms(1), 0.1)
    
#     initialized = True

#%%

# # Shortcuts
# azms = alpha_zero_move_selector_factory
# user = user_for_game(game)
# # ag = MCTSAgent(get_evaluator(net, mpdev),
# #            mcts_step_testing,
# #            move_selector_testing,
# #            temperature_testing)

#%%

# # Training Loop

# def test():
#     # Testing Settings
#     in_mcts = True
#     # MCTS
#     mcts_step_testing = 50
#     move_selector_testing = alpha_zero_move_selector_factory(1)
#     temperature_testing = 0.05
#     # Evaluator
#     move_dist_weight = 1.
#     evaluation_weight = 0.
    
#     #-------
    
#     if in_mcts:
#         agent = MCTSAgent(
#                        get_evaluator(net, mpdev),
#                        mcts_step_testing,
#                        move_selector_testing,
#                        temperature_testing)
#     else:
#         agent = EvaluatorAgent(get_evaluator(net, mpdev),
#                                       distribution_weight=move_dist_weight,
#                                       evaluation_weight=evaluation_weight)
    
#     # Performance Tracking
    
#     print('\n'+str(compare(game, [agent, UCTAgent(1000)], 20)))
#     print('\n'+str(compare(game, [agent, UCTAgent(100)], 20)))
#     print('\n'+str(compare(game, [agent, RandomAgent()], 50)))


# # Training Loop Settings
# loop_count = 30
# #===================

# for i in range(1, 1 + loop_count):
#     print("\nLoop number {}/{} is starting...\n".format(i, loop_count))
    
#     # Self-Play Settings
#     mcts_step_self_play = 50
#     move_selector_self_play = alpha_zero_move_selector_factory(5)
#     temperature_self_play = 1
#     games_per_loop = 20
    
#     # Self-Play
#     net.to(device=mpdev)
#     raw_data = generate_data(game,
#                               games_per_loop,
#                               get_evaluator(net, mpdev),
#                               mcts_step_self_play,
#                               move_selector_self_play,
#                               temperature_self_play)
    
#     ##%%
    
#     # Training Settings
#     distribution_loss = mse_loss
#     evaluation_loss = mse_loss
#     epochs_per_loop = 3
#     lr = 0.05
#     momentum = 0.9
#     weight_decay = None
#     bs = 32
    
#     # Backprop training
#     data = net.translator.translate_data(raw_data, trdev)
#     net.to(device=trdev)
#     train_data(
#         net,
#         epochs_per_loop,
#         data,
#         distribution_loss,
#         evaluation_loss,
#         lr=lr,
#         momentum=momentum,
#         weight_decay=weight_decay,
#         batch_size=bs)

#%%

