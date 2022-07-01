# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 01:31:25 2022
"""

import math
import random

from .miscellaneous import after
from .game_state_memorizer import GameStateMemorizer
from .core_search_tree import CoreSearchTree, CoreSearchNode
from .search_tree import SearchTree


default_game_state_calculator_factory = GameStateMemorizer


def alpha_zero_move_selector(search_node, exploration_constant):
    
    return max(search_node.moves.keys(),
               key=after(
                   lambda move: move.action_value
                                + (exploration_constant*move.prior_probability
                                / (1 + move.visits)),
                   lambda game_move: search_node.moves[game_move]))

def prioritize_non_visited(search_node, exploration_constant):
    
    return max(search_node.moves.keys(),
               key=after(
                   lambda move: ((move.action_value
                                + (exploration_constant*move.prior_probability
                                / (1 + move.visits))) if move.visits != 0
                                else math.inf),
                   lambda game_move: search_node.moves[game_move]))

def modified_alphazero_selector(search_node, exploration_constant):
    parent_visits = 1 + sum(move.visits for move in search_node.moves.values())
    
    return max(search_node.moves.keys(),
               key=after(
                   lambda move: move.action_value
                                + (exploration_constant
                                 * move.prior_probability
                                 * math.log(parent_visits)
                                / (1 + move.visits)),
                   lambda game_move: search_node.moves[game_move]))

def UCT(search_node, exploration_constant):
    
    parent_visits = sum(move.visits for move in search_node.moves.values())
    
    return max(search_node.moves.keys(),
               key=after(lambda move: 
                         (move.action_value
                         + exploration_constant
                         * math.sqrt(math.log(1+parent_visits)
                           / move.visits))
                         if move.visits != 0 else math.inf,
                   lambda game_move: search_node.moves[game_move]))

# TODO document
def mcts(data, evaluator, times,
         move_selector,
         exploration_constant=4.1,
         temperature = 1,
         return_type = 'move'):
    """
    An implementation of the Monte Carlo tree search.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    evaluator : TYPE
        DESCRIPTION.
    times : TYPE
        DESCRIPTION.
    exploration_constant : TYPE, optional
        DESCRIPTION. The default is 1.
    move_selector : TYPE, optional
        DESCRIPTION. The default is prioritize_non_visited.
    temperature : TYPE, optional
        DESCRIPTION. The default is 1.
    return_type : TYPE, optional
        DESCRIPTION. The default is 'move'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    

    # Not exposing the inners of search_tree to the outside
    # modules is critical.
    # If we rewrite the mcts module, we will not be required to make
    # changes to other modules.
    try:
        core_tree, game_state_calculator = data._components()
    except AttributeError:
        # So data is a game_state object
        # construct the search tree using game_state
        try:
            game = data.game()
        except AttributeError:
            raise ValueError('Either supply with a search tree or a game'
                             + ' state object as the data parameter.')
        try:
            game_state_calculator = game.game_state_calculator(data)
        except AttributeError:
            # The game does not support game_state_calculator object
            game_state_calculator = default_game_state_calculator_factory(data)

        root_game_state, root_moves, root_player = (
            game_state_calculator.initialize())

        root_move_priors, root_evaluation = (
            evaluator(root_game_state, root_moves, root_player))
        
        core_tree = CoreSearchTree(root_move_priors, root_player)                    #CoreSearchTree initialization
        search_tree = SearchTree(core_tree, game_state_calculator)
    
    core_tree, game_state_calculator = search_tree._components()
    
    #We have expanded the root
    times = times - 1
    
    if times < 0:
        raise ValueError('times must be positive.')
    
    for _ in range(times):
        address, hit_ghost, game_related_info = tree_policy(
            core_tree, game_state_calculator,
            move_selector, exploration_constant)
        
        (is_game_over, game_final_evaluation,
         game_state, new_moves, new_players_turn) = game_related_info
        
        if is_game_over:
            # Evaluation is decided by the game rules.
            evaluation = game_final_evaluation
            prior_probabilities = {}
        else:
            # Evaluator decides the values
            prior_probabilities, evaluation = evaluator(
                game_state, new_moves, new_players_turn)
        
        if hit_ghost:
            extension_policy(
                *address[-1], prior_probabilities, new_players_turn)
        
        back_up_policy(address, evaluation)
        
        # if hit_ghost:
        #     evaluation = extension_policy(
        #         *address[-1], is_game_over, game_final_evaluation,
        #         game_state, evaluator, new_moves, new_players_turn)
        # else:
        #     # This node was already in the tree. The game is over as well.
        #     evaluation = game_final_evaluation
    
    if return_type == 'tree':
        return search_tree
    
    temperature_adjusted_visits = {
        game_move: math.pow(tree_move.visits, temperature)
        for game_move, tree_move in core_tree.root.moves.items()}
    
    temperaturated_sum = sum(temperature_adjusted_visits.values())
    
    move_probabilities = {
        game_move: tree_move.visits/temperaturated_sum
        for game_move, tree_move in core_tree.root.moves.items()}
    
    move_p_as_list = list(move_probabilities.items())
    
    if return_type == 'distribution':
        return move_probabilities
    
    if return_type == 'move':
        # The code below is replaced by random library. The problem
        # is that numpy tries to create multidimensional arrays
        # when the move is coded as a tuple.
        # return _rng.choice(
        #     np.array(list(map(lambda x: x[0], move_p_as_list)),
        #                   dtype = 'object'),
        #     p = np.array(list(map(lambda x: x[1], move_p_as_list))))
        return random.choices(
            list(map(lambda x: x[0], move_p_as_list)),
            weights=list(map(lambda x: x[1], move_p_as_list))
            ).pop()
        
    
    raise ValueError('Unknown return type ' + str(return_type))

def tree_policy(core_tree, game_state_calculator,
                move_selector, exploration_constant):
    address = []
    current = core_tree.root
    game_iterator = game_state_calculator.move_iterator()
    while True:
        if len(current.moves) == 0:
            # No valid moves; the game is over.
            address.append((current, None, None))
            hit_ghost = False
            break
        
        move = move_selector(current, exploration_constant)
        outcome = game_iterator.next_outcome(move)
        address.append((current, move, outcome))
        
        tree_move = current.moves[move]
        try:
            current = tree_move.outcome_dict[outcome]
        except KeyError:
            # The leading node is not yet in the tree.
            hit_ghost = True
            break
    return address, hit_ghost, game_iterator.terminate()


def extension_policy(node, move, outcome,
                     prior_probabilities, new_players_turn):
    
    node.moves[move].outcome_dict[outcome] = CoreSearchNode(
        prior_probabilities, new_players_turn)
    

def back_up_policy(address, evaluation):
    for node, game_move, outcome in reversed(address):
        if game_move is not None:
            move = node.moves[game_move]
            #Update the action value according to the player's 
            #expected gain determined by the lookahead.
            player = node.player
            move.action_value = (move.action_value*move.visits +
                        evaluation[player])/(move.visits + 1)
            move.visits = move.visits +1
            
            
            
