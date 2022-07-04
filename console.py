"""
This module is intended for development purposes.


PROJECT COMMITMENTS (FORMAT THIS LATER ON):
    
    --> There are no valid moves if and only if game is over.
    
    --> GameState should provide hashable players. this was first
    needed in game() function to check whether all players
    are present.
    
    --> Reversi game state is immutable and supports
    __eq__ for state comparison
    
    --> NN module should provide a function for creating
    evaluators from nns. signature: as_evaluator(nn)
    
PROJECT BUG WARNINGS:
    
    --> numpy.array function tries to create multidimensional
    arrays when fed list of tuples. This is a problem because
    we use random sampling on moves and they can indeed be
    tuples. This results in unexpected behaivour in MCTS.
    This is a problem in mcts_logic when picking a random
    move from the mcts distribution and also in 
    game_state_memorizer when picking a random outcome.
    
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

from mcts import *
from game.tictactoe import TicTacToe as tct
from game.reversi import Reversi
from training_tools import train, TournamentGameSession
from neural_network import TicTacToe_defaultNN as NN
from training_tools import TrainingGameSession as TGS
from training_tools import TournamentGameSession as ToGS
from game_session import GameSessionTemplate
from agent import agents

uct_agent = agents.UCTAgent(10000, temperature = 1)
random_agent = agents.RandomAgent()

# net = NN('cpu')
# trd, ted, vad = load_UCT_data(net)

class ConsoleGameSession(GameSessionTemplate):
    
    def __init__(self, state, agent_dict, delay = 1):
        self._state = state
        self._agent_dict = agent_dict
        self._delay = delay
        self._history = [self._state]
        
    def current_state(self):
        return self._state
    
    def proceed(self, player):
        time.sleep(self._delay)
        agent = self._agent_dict[player]
        print('Agent {} as {} is thinking...\n'.format(
            str(agent), str(player)))
        move = agent.ask(self.current_state())
        if move not in self.current_state().moves():
            raise RuntimeError('Agent {} committed an illegal move...'.format(
                agent))
        self._state = self._state.after(move, None)
        print('Agent {} has played {}\n'.format(str(agent), str(move)))
        print('Current Game State:\n{}'.format(str(self._state)))
        self._history.append(self._state)
    
    def initialize(self):
        print('The game is starting...\n')
        print('The agents play as:')
        for player, agent in self._agent_dict.items():
            print('{} plays as {}'.format(str(agent), str(player)))
    
    def finalize(self):
        print('The game has ended.\n{}\n'.format(str(self._state)))
        print('''The results are: 
              \n{}'''.format(str(self._state.game_final_evaluation())))
        print('The agents play as:')
        for player, agent in self._agent_dict.items():
            print('{} plays as {}'.format(str(agent), str(player)))
    
    def return_value(self):
        return self._history
    
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
            (neural_network.state_to_input(state),
             *neural_network.translate_out(move_distribution, evaluation))
            for state, move_distribution, evaluation in a_data]
        new_a_data = []
        for i in range(3):
            new_a_data.append(torch.cat(
                [point[i].unsqueeze(dim=0) for point in a_data],
                dim=0))
        new_data.append(new_a_data)
    
    return tuple(new_data)



# def as_direct_player(evaluator):
#     return lambda state: random.choices(*zip(*evaluator(
#         state, state.moves(), state.turn())[0].items()))[0]


# def as_mcts_player(evaluator, mcts_steps, move_selector, exploration_constant,
#               temperature):
    
#     return lambda state: mcts(state, evaluator, mcts_steps, move_selector,
#                               exploration_constant, temperature,
#                               return_type = 'move')


def supervised_train(net, epochs):
    
    device = 'cpu'
    
    training_data, test_data, validation_data = load_UCT_data(net)
    
    x_data, dist_data, eval_data = training_data
    
    move_loss_function = torch.nn.functional.cross_entropy
    eval_loss_function = torch.nn.functional.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    batch_size = 32
    
    num_of_batches = math.ceil(len(x_data)/32) * epochs
    
    # start training
    net.train()
    
    for _ in range(num_of_batches):
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
        
        print("Mb complete. TLoss is: {}, MLoss is: {} ELoss is: {}".format(loss.item(), dist_loss.item(), eval_loss.item()))
    
    net.eval()


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
def play_tct():
    # Initialize the game
    ga = tct.initial_state()
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
        move = mcts(ga, random_playout_evaluator,
                    10000, UCT_move_selector,
                    4.1, 1)
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
            match = re.match('\d\d', in_)
            if match is None:
                if in_ == 'e':
                    globals()['last_game'] = ga
                    return
                if in_ == 'p':
                    break
                print('Prompt not recognized.')
                continue
            x,y = map(int, match[0])
            x,y = x-1,y-1
            if (x,y) in map(lambda x: x[:-1],ga.moves()):
                break
            else:
                print('Move is not possible.')
                continue
        # update the game with the move
        ga = ga.after((x, y, ga.turn()),None)
        # Notify the spectator
        print('YOU PLAYED\n' + str(ga))
    
    #Notify the spectator (maybe return other data as well).
    print('\n\nGame is over!\n\nThe results are:')
    print(ga.game_final_evaluation())

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








