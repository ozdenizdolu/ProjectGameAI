import math

from miscellaneous import after
 
def alpha_zero_move_selector_factory(exploration_constant):
    def func(search_node):
        
        # Do not add one, this is exactly the sum of the children visits
        # unlike UCT
        parent_visits = (
            sum(move.visits for move in search_node.moves.values()))
        
        return max(search_node.moves.keys(),
                   key=after(lambda move: 
                        # Action Value
                        move.action_value
                        # Plus upper confidence bound
                        + (exploration_constant
                        * move.prior_probability
                        * math.sqrt(parent_visits)
                        / (1 + move.visits)),
                             lambda game_move: search_node.moves[game_move]))
    return func

def UCT_move_selector_factory(exploration_constant):
    def func(search_node):
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