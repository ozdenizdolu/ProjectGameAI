import math

import numpy as np
from numpy.random import default_rng

from miscellaneous import after

rng = default_rng()
 
def alpha_zero_move_selector_factory(exploration_constant):
    def func(search_node, dirichlet_noise_parameter, noise_contribution):
        
        # Do not add one, this is exactly the sum of the children visits
        # unlike UCT
        d = lambda game_move: search_node.moves[game_move]
        
        parent_visits = (
            sum(move.visits for move in search_node.moves.values()))
        
        if dirichlet_noise_parameter == noise_contribution == None:
            adjusted_prior = {move: d(move).prior_probability
                              for move in search_node.moves.keys()}
        else:
            noise_array = rng.dirichlet(
                [dirichlet_noise_parameter]*len(search_node.moves))
            noise = {move:val for move,val in zip(search_node.moves.keys(),
                                                  noise_array)}
            
            adjusted_prior = {
                move: ((1-noise_contribution)*d(move).prior_probability
                       + noise_contribution*noise[move])
                for move in search_node.moves.keys()}
        
        return max(search_node.moves.keys(),
                   key = lambda move: 
                        # Action Value
                        d(move).action_value
                        # Plus upper confidence bound
                        + (exploration_constant
                        # Where we use noise adjusted priors
                        * adjusted_prior[move]
                        * math.sqrt(parent_visits)
                        / (1 + d(move).visits)))
    func.exploration_constant = exploration_constant
    return func


# Do not use because dirichlet noise interface is not added
def UCT_move_selector_factory(exploration_constant):
    def func(search_node, dirichlet_noise_parameter, noise_contribution):
        
        if dirichlet_noise_parameter == noise_contribution == None:
            pass
        else:
            raise NotImplementedError()
        # Add one to make it positive.
        parent_visits = (
            1 + sum(move.visits for move in search_node.moves.values()))
        
        return max(search_node.moves.keys(),
                    key=after(lambda move: 
            (move.action_value
            + (2 * exploration_constant
            * math.sqrt(math.log(parent_visits)
            / move.visits)))
            if move.visits != 0 else math.inf,
                        lambda game_move: search_node.moves[game_move]))
    func.exploration_constant = exploration_constant
    return func


# This generalization in unnecessary. Confidence bound calculation differs
# and it is better to write the whole function.
# def get_confidence_bound_move_selector(confidence_bound):
#     """
#     Returns a move selector which selects according to Q + U where
#     Q is the action value of the move and U is the confidence bound.
    
#     Parameters
#     ----------
#     confidence_bound : callable
#         This takes in CoreSearchNode and outputs a number which is interpreted
#         to be the upper confidence bound of the node.

#     Returns
#     -------
#     callable
#         returns the move selector.

#     """