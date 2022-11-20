"""
This module is intended for development purposes. Changes to this module
only represent some of the experiments done at the time of the change,
and should not be taken as a part of the codebase.


PROJECT COMMITMENTS (FORMAT THIS LATER ON):
    
    --> There are no valid moves if and only if game is over.
    
    --> GameState should provide hashable players. this was first
    needed in game() function to check whether all players
    are present.
    
    --> Reversi game state is immutable and supports
    __eq__ for state comparison
    
    --> NN module should provide a function for creating
    evaluators from nns. signature: as_evaluator(nn)
    
    --> Agents should support being keys in a dictionary. Trivial
    way to ensure this is to not extend __eq__ and __hash__ and
    be content with == being is.
    
PROJECT BUG WARNINGS:
    
    --> numpy.array function tries to create multidimensional
    arrays when fed list of tuples. This is a problem because
    we use random sampling on moves and they can indeed be
    tuples. This results in unexpected behaivour in MCTS.
    This is a problem in mcts_logic when picking a random
    move from the mcts distribution and also in 
    game_state_memorizer when picking a random outcome.
    
    --> Torch.nn.functional.cross_entropy is not what it
    claims to be! Use the project's cross entropy function
    instead.
    
CHANGES:
    
    --> Outcomes should no longer return numpy arrays. It should
    return plain lists. This change was made due to numpy.ndarray's
    weird behaivour on 1d arrays of tuples. When moves are tuples as
    well, then numpy creates a 2d array instead of 1d array of moves.
    
HIGH LEVEL NOTES:
    
    --> Use noise on prior probabilities for stimulating exploration.
    This is what is done with AlphaZero. (This is scaled for games
    according to their number of legal moves in an average position).
    
    --> Use Bayesian methods for hyperparameter optimization.
    
    --> For performance, it is possible to use the same tree for
    the generation of subsequent moves during training game generation.
    I decided not to implement it yet due to biasses introduced.
    
    
"""


#Tools for the interactive session

import re
import random
from importlib import reload
import cProfile
import time
import itertools
from collections import Counter
import pickle
import math

import numpy as np
import torch
from torch.nn.functional import mse_loss # Do not use cross entropy of torch's
import matplotlib
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

from mcts import *
from game.tictactoe import TicTacToe as tct
from game.reversi import Reversi
from training_tools import unsupervised_training, TournamentGameSession
from neural_network import (TicTacToe_defaultNN,
                            TowardsAttention,
                            NetworkWithAttention,
                            MultiHeadSelfAttentionBlock,
                            NetworkWithMultiHeadAttention)
from training_tools import TournamentGameSession as ToGS
from game_session import GameSessionTemplate
from agent import agents
from training_tools import compare
from miscellaneous import cross_entropy
from network_translator import (StandardTicTacToeTranslator,
                                TokenLikeTicTacToeTranslator)

EvaluatorAgent = agents.EvaluatorAgent
RandomAgent = agents.RandomAgent
UCTAgent = agents.UCTAgent
MCTSAgent = agents.MCTSAgent

uct_agent = agents.UCTAgent(10000, temperature = 1)
good_uct_agent = agents.UCTAgent(10000, temperature = 0.1)
random_agent = agents.RandomAgent()


device = 'cpu'

net = NetworkWithMultiHeadAttention(128, 16, device)
# net = TowardsAttention(10, device)
net.eval()
tr = TokenLikeTicTacToeTranslator()

if 'prev_loss' not in vars():
    prev_loss = ([],[])

with open('tct_all_UCT_data.pickle','rb') as file:
    raw_data = pickle.load(file)
    random.shuffle(raw_data)
    
    training_data = raw_data[0:15000]
    test_data = raw_data[15000:18000]
    validation_data = raw_data[18000:]
    
    trd = tr.translate_data(training_data, device)
    ted = tr.translate_data(test_data, device)
    vad = tr.translate_data(validation_data, device)
    
# with open('network_prepared_data_temp.pickle','rb') as file:
#     trd, ted, vad = pickle.load(file)

def discretize_tct_answer(move_dist, evaluation):
    
    selected_move = max(move_dist, key = lambda move: move_dist[move])
    
    #THIS DOESNT WORK. 0 evaluation should say the game is draw
    #The game outcome is not always a win!!
    # predicted_winner = max(evaluation, key = lambda player: evaluation[player])
    
    x_bets = evaluation[tct.X]
    
    if x_bets > 0.5:
        pred_result = 1
    elif x_bets < -0.5:
        pred_result = -1
    else:
        pred_result = 0
    
    return selected_move, pred_result

def discrete_tct_answer(evaluator, state):
    """
    Discretizes the evalautor's assestment about the state.

    Parameters
    ----------
    evaluator : evaluator
        
    state : game state
        

    """
    return discretize_tct_answer(*evaluator(state))
    
    
def calculate_loss(network, data, dist_loss_fn, eval_loss_fn):
    """
    data is a triplet (x, dist_target, eval_target)
    """
    
    x, dist_target, eval_target = data
    
    dist_net, eval_net = network(x)
    
    return (dist_loss_fn(dist_net, dist_target),
            eval_loss_fn(eval_net, eval_target))


def generate_all_UCT_data(mcts_steps, temp,
                          stop_at = math.inf, results = None):
    if results is None:
        results = {}
    all_states = tct.all_states()
    # random.shuffle(all_states) not shuffling is better to see empty pos first
    for state in all_states:
        if len(results) > stop_at:
            break
        if state in results:
            continue
        # Not previously explored
        data_to_end = []
        TGS(state, random_playout_evaluator, data_to_end,
            mcts_steps, 4.1, UCT_move_selector, temp).run()
        for state, dist, evaluation in data_to_end:
            if state not in results:
                results[state] = (dist, evaluation)
        print('''An iteration complete, {} new(!) states, total of {} states.
              '''.format(len(data_to_end), len(results)))
    
    return [(state, distribution, evaluation)
                for state, out in results.items()
                for distribution, evaluation in [out]]

def generate_UCT_training_data(num_of_games, mcts_steps_per_move, temperature):
    training_data = []
    for i in range(num_of_games):
        state = tct.random_state()
        TGS(state, random_playout_evaluator, training_data,
            mcts_steps_per_move, 4.1, UCT_move_selector, temperature).run()
        print('{} / {} is comlete'.format(i, num_of_games))
    return training_data


def load_UCT_data(neural_network = None):
    """
    Loads pre-prepared data generated using UCT for the tictactoe game.

    Parameters
    ----------
    neural_network : neural network, optional
        If a neural network which has compatible io with the tictactoe game
        is provided, then the data is transformed into PyTorch tensors
        suitable for being input and output to the network is returned.

    Returns
    -------
    data : tuple
        length of the tuple is 3. First element is training data, second
        is test data and the third is validation data for the dataset.
        Each data is a list of (state, distribution, evaluation) triples.
        If neural network is provided then each data is a triplet of
        tensors whose first dimension corresponds to the index of 
        training examples.

    """
    with open('UCT_data.pickle', 'rb') as file:
        data = pickle.load(file)
        
    if neural_network is None:
        return data
    
    new_data = []
    
    for a_data in data:
        a_data = [
            (neural_network.states_to_tensor([state]),
             neural_network.dists_to_tensor([move_distribution]),
             neural_network.evals_to_tensor([evaluation]))
            for state, move_distribution, evaluation in a_data]
        new_a_data = []
        for i in range(3):
            new_a_data.append(torch.cat(
                [point[i] for point in a_data],
                dim=0))
        new_data.append(new_a_data)
    
    return tuple(new_data)



#TODO: refactor this
def supervised_train(net, epochs, data,
                     move_loss_function,
                     eval_loss_function,
                     lr=0.01, momentum = 0.9, weight_decay=0.,
                     batch_size=32,
                     testing_data = None,
                     generate_plot_epoch = 5,
                     previous_losses = None):
    
    if previous_losses is not None:
        trd_loss_tracker, vad_loss_tracker = previous_losses
    else:
        trd_loss_tracker = []
        vad_loss_tracker = []
        
    if weight_decay != 0.:
        raise NotImplementedError('''Weight decay currently applies to all
                                  parameters, not just weights. Update the
                                  code to use this functionality securely.''')
    
    
    x_data, dist_data, eval_data = data
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    
    num_of_batches_per_epoch = math.ceil(len(x_data)/batch_size)
    
    # start training
    net.eval()
    print('Training is starting...\n')
    print('Training Data Performance:\n')
    print(performance_on_data(net, trd,
                              move_loss_function, eval_loss_function,
                              record = trd_loss_tracker))
    
    if testing_data != None:
        print('Test Data Performance:\n')
        print(performance_on_data(net, testing_data,
                                  move_loss_function, eval_loss_function,
                                  record = vad_loss_tracker))
    
    net.train()
    
    for epoch in range(1, epochs+1):
        for _ in range(num_of_batches_per_epoch):
            example_indices = torch.tensor(random.sample(
                range(x_data.shape[0]), batch_size))
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
        print('Epoch {}/{} complete.\n'.format(epoch,epochs))
        print('Training Data Performance:\n')
        print(performance_on_data(net, data,
                                  move_loss_function, eval_loss_function,
                                  record = trd_loss_tracker))
        if testing_data != None:
            print('Test Data Performance:\n')
            print(performance_on_data(net, testing_data,
                                      move_loss_function, eval_loss_function,
                                      record = vad_loss_tracker))
        net.train()
        
        if epoch % generate_plot_epoch == 0 or epoch == 1:
            tr_dist, tr_eval = zip(*trd_loss_tracker)
            va_dist, va_eval = zip(*vad_loss_tracker)
            
            tr_dist_min = move_loss_function(
                data[1],data[1])
            # tr_eval_min = eval_loss_function(, )
            va_dist_min = move_loss_function(
                testing_data[1],testing_data[1])
            # va_eval_min = eval_loss_function(, )
            
            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.plot(list(range(len(trd_loss_tracker))), [tr_dist_min]*len(trd_loss_tracker), color = 'r')
            ax1.plot(list(range(len(trd_loss_tracker))), [va_dist_min]*len(trd_loss_tracker), color = 'g')
            ax1.plot(list(range(len(trd_loss_tracker))), tr_dist, color='r')
            ax2.plot(list(range(len(trd_loss_tracker))), tr_eval, color='m')
            ax1.plot(list(range(len(vad_loss_tracker))), va_dist, color='g')
            ax2.plot(list(range(len(vad_loss_tracker))), va_eval, color='b')
            # ax.set_ylim(top = )
            plt.show()
    net.eval()

@torch.no_grad()
def performance_on_data(net, data,
                        move_loss_function,
                        eval_loss_function,
                        record = None):
    x_data, dist_data, eval_data = data
    
    net_dist, net_eval = net(x_data)
    
    output = ''
    output += 'Distribution loss is {}\n'.format(
        move_loss_function(net_dist, dist_data))
    output += 'Minimum Distribution loss is {}\n\n'.format(
        move_loss_function(dist_data, dist_data))
    
    output += 'Evaluation loss is {}\n'.format(
        eval_loss_function(net_eval, eval_data))
    output += 'Minimum Evaluation loss is {}\n'.format(
        eval_loss_function(eval_data, eval_data))
    
    if record is not None:
        record.append((move_loss_function(net_dist, dist_data).item(),
                       eval_loss_function(net_eval, eval_data).item()))
    
    return output
    

def get_evaluator(network, translator, device = 'cpu'):
    def evaluator(state, legal_moves = None, player = None):
        if legal_moves == None:
            legal_moves = state.moves()
        if player == None:
            player = state.turn()
        x = translator.states_to_tensor([state], device = device)
        move_dist, evaluation = network(x)
        return (
            translator.tensor_to_dists([state], [legal_moves], move_dist)[0], 
            translator.tensor_to_evals([state], evaluation)[0]
            )
    return evaluator

# a = tct.initial_state()
# a = a.after((1,1,100),None)
# a = a.after((0,0,101),None)
# print(a)
# a = a.after((0,1,100),None)
# print(a)
# a = a.after((1,0,101),None)
# print(a)

# tree =mcts(a, random_playout_evaluator, 10000, UCT_move_selector, 4.1, 1, return_type = 'tree')

# t = tree._core_tree.as_treelib_tree()

# if __name__ == '__main__':
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu"
#     # net = NN(device).to(device)
    
#     # train(tct, net, 100, modified_alphazero_selector, 100, 4, 160, 1.0, 10, 5e-2, 1, 4, device)
    
#     net = torch.load('checkpointfinal.pt')
    
#     net_player = lambda state: mcts(state, net.as_evaluator(), 100,
#                 modified_alphazero_selector)
    
#     random_player = lambda state: random.choice(state.moves())
    
#     UCT_player = lambda state: mcts(state, random_playout_evaluator, 100,
#                                     modified_alphazero_selector)
    
    
#     results = [{} for _ in range(100)]
    
#     for result in results:
#        # TournamentGameSession(tct, {tct.O:net_player,# changed the function
#                                    # tct.X:UCT_player}, result).run()
    
#     print('O = ' +str(sum(result[tct.O]>0.5 for result in results)))
#     print('X = ' +str(sum(result[tct.X]>0.5 for result in results)))


# TODO
# def compare(players, times):
#     """Distribute player's colours and compare their payoffs."""
#     player_dict = {i: player for i,player in enumerate(players)}
    
    

#Temporary
def request_move_from_console(state, move_interpreter):
    while True:
        in_ = input()
        try:
            move = move_interpreter(state, in_)
            if move in state.moves():    
                break
            else:
                print('\nMove is not applicable in this position.')
        except ValueError as e:
            print(str(e))
    return move
    
#Temporary
# def play_tct():
#     # Initialize the game
#     ga = tct.initial_state()
#     # Notify the spectator
#     print('\n\n')
#     print(ga)
#     print('\n\nGame is starting')
#     # Create the mood
#     time.sleep(2)
    
#     while not ga.is_game_over():
#         # Notify the spectator about what is happening
#         print('\n\nComputer is thinking...')
#         # Receive the move from the player (maybe some more data as well)
#         move = mcts(ga, random_playout_evaluator,
#                     10000, UCT_move_selector,
#                     4.1, 1)
#         # Update the game
#         ga = ga.after(move, None)
#         # Notify the spectator
#         print('\n'+str(ga))
#         # ... not a clean code
#         if ga.is_game_over():
#             break
#         # Request move from the player
#         print('\nYour turn...')
#         while True:
#             in_ = input()
#             match = re.match('\d\d', in_)
#             if match is None:
#                 if in_ == 'e':
#                     globals()['last_game'] = ga
#                     return
#                 if in_ == 'p':
#                     break
#                 print('Prompt not recognized.')
#                 continue
#             x,y = map(int, match[0])
#             x,y = x-1,y-1
#             if (x,y) in map(lambda x: x[:-1],ga.moves()):
#                 break
#             else:
#                 print('Move is not possible.')
#                 continue
#         # update the game with the move
#         ga = ga.after((x, y, ga.turn()),None)
#         # Notify the spectator
#         print('YOU PLAYED\n' + str(ga))
    
#     #Notify the spectator (maybe return other data as well).
#     print('\n\nGame is over!\n\nThe results are:')
#     print(ga.game_final_evaluation())

#Temporary
def play_reversi(mcts_steps = 1000):
    # Initialize the game
    ga = Reversi.initial_state()
    # Notify the spectator
    print('\n\n')
    print(ga)
    print('\n\nGame is starting')
    # Create the mood
    time.sleep(2)
    
    while not ga.is_game_over():
        # Notify the spectator about what is happening
        print('\n\nComputer is thinking...')
        # Receive the move from the player (maybe some more data as well)
        move = mcts(ga, random_playout_evaluator, mcts_steps)
        # Update the game
        ga = ga.after(move, None)
        # Notify the spectator
        print('\n'+str(ga))
        # ... not a clean code
        if ga.is_game_over():
            break
        # Request move from the player
        print('\nYour turn...')
        while True:
            in_ = input()
            #TODO include the pass move
            match = re.match('\d\d', in_)
            if match is None:
                if in_ == 'e':
                    break
                if in_ == 'pass' and Reversi.PASS_MOVE in ga.moves():
                    move = Reversi.PASS_MOVE
                    break
                print('Prompt not recognized.')
                continue
            x,y = map(int, match[0])
            x,y = x-1,y-1
            if (x,y) in ga.moves():
                move = (x,y)
                break
            else:
                print('Move is not possible.')
                continue
        # update the game with the move
        ga = ga.after(move, None)
        # Notify the spectator
        print('YOU PLAYED\n' + str(ga))
    
    #Notify the spectator (maybe return other data as well).
    print('\n\nGame is over!\n\nThe results are:')
    print(ga.game_final_evaluation())








