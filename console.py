# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 02:21:30 2022


PROJECT COMMITMENTS (FORMAT THIS LATER ON):
    
    --> There are no valid moves if and only if game is over.
    
    --> GameState should provide hashable players. this was first
    needed in game() function to check whether all players
    are present.
    
    --> Reversi game state is immutable and supports
    __eq__ for state comparison
    
    --> NN module should provide a function for creating
    evaluators from nns. signature: as_evaluator(nn, game)
    
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
    return plain lists. This change was made due to the numpy.array
    weird behaivour on 1d arrays of tuples.
    
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


import numpy as np
import torch


from mcts import *
from game.tictactoe import TicTacToe as tct
from game.reversi import Reversi
from training_tools import train, TournamentGameSession
from neural_network import TicTacToe_defaultNN as NN

if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # net = NN(device).to(device)
    
    # train(tct, net, 100, modified_alphazero_selector, 100, 4, 160, 1.0, 10, 5e-2, 1, 4, device)
    
    net = torch.load('checkpointfinal.pt')
    
    net_player = lambda state: mcts(state, net.as_evaluator(), 100,
                modified_alphazero_selector)
    
    random_player = lambda state: random.choice(state.moves())
    
    UCT_player = lambda state: mcts(state, random_playout_evaluator, 100,
                                    modified_alphazero_selector)
    
    
    results = [{} for _ in range(100)]
    
    for result in results:
        TournamentGameSession(tct, {tct.O:net_player,
                                    tct.X:UCT_player}, result).run()
    
    print('O = ' +str(sum(result[tct.O]>0.5 for result in results)))
    print('X = ' +str(sum(result[tct.X]>0.5 for result in results)))


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
        move = mcts(ga, random_playout_evaluator, 10000)
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
