"""
ProjectGameAI
=============

Provides reinforcement learning tools such as

- Monte Carlo tree search (MCTS)
- Upper confidence bounds applied to trees (UCT) algorithm
- Implementation of games reversi and tictactoe for experimentation.

example code
------------

import ProjectGameAI as ai

state = ai.TicTacToe.initial_state()
# Run the UCT algorithm with 10000 random playouts.
move = ai.uct(state, times=10000, exploration_constant=1., temperature=0.1)
# apply the move in the game. 
state = state.after(move, outcome=None)
print(state)
# Play O to upmost place.
state = state.after((0,1,ai.TicTacToe.O), outcome=None)
# Ask UCT for a move.
move = ai.uct(state, times=10000, exploration_constant=1.)
state = state.after(move, outcome=None)
print(state)


example code 2
--------------

import ProjectGameAI as ai

# Create a game session where two UCT algorithms compete.
ses = ai.ConsoleGameSession(ai.TicTacToe.initial_state(), 
                            {ai.TicTacToe.X: ai.MCTSAgent(
                                ai.random_playout_evaluator, 10000), 
                            ai.TicTacToe.O: ai.UCTAgent(10000)})
ses.run()

"""

from .mcts import mcts, uct
from .game_session import ConsoleGameSession
from .game import Reversi, TicTacToe
from .evaluator import random_playout_evaluator
from .agent import (EvaluatorAgent, MCTSAgent, RandomAgent, 
                    UCTAgent, console_user_for_game)