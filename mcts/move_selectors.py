import math

from miscellaneous import after

def alpha_zero_move_selector_function(exploration_constant):
    def func(search_node):
        return max(search_node.moves.keys(),
                    key=after(
                        lambda move: move.action_value
                                    + (exploration_constant*move.prior_probability
                                    / (1 + move.visits)),
                        lambda game_move: search_node.moves[game_move]))
    return func


def prioritize_non_visited_move_selector_factory(exploration_constant):
    def func(search_node):
        return max(search_node.moves.keys(),
                    key=after(
                        lambda move: ((move.action_value
                                    + (exploration_constant*move.prior_probability
                                    / (1 + move.visits))) if move.visits != 0
                                    else math.inf),
                        lambda game_move: search_node.moves[game_move]))
    return func


# def modified_alphazero_selector_factory(exploration_constant):
#     def func(search_node):
#         parent_visits = 1 + sum(move.visits for move in search_node.moves.values())
        
#         return max(search_node.moves.keys(),
#                     key=after(
#                         lambda move: move.action_value
#                                     + (exploration_constant
#                                       * move.prior_probability
#                                       * math.log(parent_visits)
#                                     / (1 + move.visits)),
#                         lambda game_move: search_node.moves[game_move]))
#     return func

def UCT_move_selector_factory(exploration_constant):
    def func(search_node):
        
        parent_visits = (
            1 + sum(move.visits for move in search_node.moves.values()))
        
        return max(search_node.moves.keys(),
                    key=after(lambda move: 
                              (move.action_value
                              + 2*exploration_constant
                              * math.sqrt(math.log(parent_visits)
                                / move.visits))
                              if move.visits != 0 else math.inf,
                        lambda game_move: search_node.moves[game_move]))
    return func
