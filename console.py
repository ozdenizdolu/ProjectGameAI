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
    
    
"""

import re
#for the console
from importlib import reload

# from game.reversi.reversi_game_state import ReversiGameState
# from game.reversi.reversi_game_state import random_play


from mcts import *
import game.tictactoe.tictactoe as tct
import time




# def play_anything(game_state):
    
#     g = game_state
    
#     for spectator in spectators:
#         notify spectator

# Want a function which makes it play any game
# Want to make ai compete as well as me in console or screen
# Want to store extra information and send it during execution
# Want support for several player games

def game_session(initialization_strategy, move_strategy,
                 finalization_strategy):
    
    initialization_strategy()
    while not game.is_game_over():
        move_strategy()
    finalization_strategy()



# def game_session(agents, dataholder):
    
    
    


def game(game, players):
    g = [game.initial_state()]
    m = []
    o = []
    
    while not g[-1].is_game_over():
        m.append(players[g[-1].turn].request_move())
        o.append(outcome = g[-1].outcomes(move, pick_one = True))
        g.append(g[-1].after(m[-1], o[-1]))
        
    # At this point g has one more element than others
    
    return g[-1].game_final_evaluation()

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








# state = ReversiGameState.initial_state()

# tree = mcts(state, random_playout_evaluator, 100, exploration_constant= 50,
#             return_type = 'tree')


# def play():
#     # Play a reversi game with modified UCT
#     state = ReversiGameState.initial_state()
#     print('Initialized.\n'+str(state))
#     while not state.is_game_over():
#         ai_move = mcts(
#             state, random_playout_evaluator,
#             1000, exploration_constant= 50,
#             return_type = 'move')
#         state = state.after(ai_move, None)
#         print(str(state)+'\nYour turn')
#         while True:
#             in_ = input()
#             match = re.match('\d\d', in_)
#             if match is None:
#                 print('Prompt not recognized.')
#                 continue
#             x,y = map(int, match[0])
#             x,y = x-1,y-1
#             if (x,y) in state.moves():
#                 break
#             else:
#                 print('Move is not possible.')
#                 continue
#         state = state.after((x,y),None)
#         print('YOU PLAYED\n' + str(state))
        
        
# play()

# def game(player_black, player_white):
#     state = ReversiGameState.initial_state()
    
#     #TODO
        



