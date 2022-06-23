# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:15:49 2022
"""

import random

def random_playout_evaluator(game_state, moves, player):
    prior_probabilities = {move: 1/len(moves) for move in moves}
    current_state = game_state
    while not current_state.is_game_over():
        move = random.choice(list(current_state.moves()))
        current_state = current_state.after(move, None)
    return prior_probabilities, current_state.game_final_evaluation()

