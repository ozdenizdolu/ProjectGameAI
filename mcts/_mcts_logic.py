import math
import random

from numpy.random import default_rng

from miscellaneous import after, PDist
from .game_state_calculators._game_state_memorizer import GameStateMemorizer
from ._core_search_tree import CoreSearchTree, CoreSearchNode
from .search_tree import SearchTree
from .move_selectors import alpha_zero_move_selector_factory



rng = default_rng()


# This is used when the game do not support a special game state calculator.
_default_game_state_calculator_factory = GameStateMemorizer


_default_move_selector_factory = (
    lambda: alpha_zero_move_selector_factory(4))

#TODO: Implement noise properly so that the same noise is used.
# If a tree is reused then the noise should be handled properly.

#TODO: Did I prevent a bug where the evaluator does return an invalid move?
# Does game state calculator catch that? --> I don't think so.
# The evaluator may be giving an invalid move and the search may consider it.
# However if the search goes through there, a key error will be raised in
# game state calculator.
def mcts(data,
         evaluator,
         times,
         move_selector = None,
         temperature = 1,
         dirichlet_noise_parameter = None,
         noise_contribution = None,
         return_type = 'move'):
    """
    An implementation of the Monte Carlo tree search.

    Parameters
    ----------
    data : game state or a search tree
        data is the data on which mcts search will be conducted. If
        it is a game state, then a new search tree is initialized
        at that state as the root. If it is a search tree then the
        search will be continued on that tree.
        
    evaluator : evaluator
        The evaluator which will lead the search. See glossary for
        the definition of evaluator.
        
    times : positive integer
        If data is a game state then times is the number of nodes of
        the search tree at the end of the algorithm, excluding the root.
        If data is a search tree then times is the number of nodes added
        to the search tree.
        
    move_selector : move selector
        A move selector provided by the mcts package. If not provided
        then the default one will be used.
        
    temperature : float
        Temperature is used to process the raw visit counts to probabilities.
        Low temperature prefers the best nodes. Higher the temperature,
        more uniform the distribution is.
    
    return_type : string
        This is either 'move', 'tree', or 'distribution'. If 'move' then
        the move selected according to temperature and the search will
        be returned. If 'distribution' then the distribution on the moves
        will be returned as a miscellaneous.PDist.
        If 'tree', then the search tree will be returned.

    Returns
    -------
        Depends on return_type.

    """
    
    # Not exposing the inners of search_tree to the outside
    # modules is critical.
    # If we rewrite the mcts module, we will not be required to make
    # changes to other modules.
    
    if isinstance(data, SearchTree):
        search_tree = data
        if not (dirichlet_noise_parameter == noise_contribution == None):
            raise ValueError('''A previous search is used again with
                             Dirictlet noise. This is not supported
                             to avoid silent bugs.''')
    else:
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
            game_state_calculator = _default_game_state_calculator_factory(data)
            
        root_game_info = game_state_calculator.request_root()
        
        if root_game_info.is_game_over:
            raise ValueError(
                'MCTS search cannot be done on a finished game state.')
        
        raw_root_move_priors, root_evaluation = evaluator(
                                             root_game_info.game_state,
                                             root_game_info.legal_moves,
                                             root_game_info.turn)
        
        if raw_root_move_priors.keys() != root_game_info.legal_moves:
            raise ValueError('''The evaluator failed to give the 
                             legal moves in its distribution at the root.
                             {} , {}'''.format(raw_root_move_priors,
                             root_game_info.legal_moves))
        
        if dirichlet_noise_parameter == None:
            root_move_priors = raw_root_move_priors
            
        else:
            noise_array = rng.dirichlet(
                [dirichlet_noise_parameter]*len(raw_root_move_priors.keys())
                )
            
            noise = {move: val for move,val in zip(raw_root_move_priors.keys(),
                                                  noise_array)}
            
            #Added noise to root priors similar to the AlphaZero article.
            root_move_priors = {
                    move: ((1-noise_contribution)*raw_root_move_priors[move]
                           + noise_contribution*noise[move])
                    for move in raw_root_move_priors.keys()}
        
        core_tree = CoreSearchTree(root_move_priors, root_game_info.turn)
        
        search_tree = SearchTree(core_tree, game_state_calculator)
    
    core_tree, game_state_calculator = search_tree._components()

    if move_selector == None:
        move_selector = _default_move_selector_factory()

    if times < 0:
        raise ValueError('times must be positive.')
        
    

    # The main loop of the algorithm.

    for _ in range(times):
        address, hit_ghost, new_game_state_info = _tree_policy(
            core_tree, game_state_calculator,
            move_selector)

        if new_game_state_info.is_game_over:
            # Evaluation is decided by the game rules.
            evaluation = new_game_state_info.game_final_evaluation
            prior_probabilities = {}
        else:
            # Evaluator decides the values
            prior_probabilities, evaluation = evaluator(
                new_game_state_info.game_state,
                new_game_state_info.legal_moves,
                new_game_state_info.turn)

        if hit_ghost:
            _extension_policy(*address[-1],
                             prior_probabilities,
                             new_game_state_info.turn)

        _back_up_policy(address, evaluation)

    # Provide the requested information to the client:

    if return_type == 'tree':
        return search_tree
    
    temperature_adjusted_visits = {
        game_move: math.pow(tree_move.visits, (1 / temperature))
        for game_move, tree_move in core_tree.root.moves.items()}
    
    temperaturated_sum = sum(temperature_adjusted_visits.values())
    
    move_probabilities = {
        game_move: temperature_adjusted_visits[game_move]/temperaturated_sum
        for game_move, tree_move in core_tree.root.moves.items()}
    
    # move_p_as_list = list(move_probabilities.items())
    
    move_probabilities_as_PDist = PDist(*zip(*move_probabilities.items()),
                                    normalize = False, supports_hash= True)
    
    if return_type == 'distribution':
        return move_probabilities_as_PDist
    
    if return_type == 'move':
        # The code below now uses random library instead of numpy.
        # The problem is that numpy tries to create multidimensional arrays
        # when the move is coded as a tuple.
        # return _rng.choice(
        #     np.array(list(map(lambda x: x[0], move_p_as_list)),
        #                   dtype = 'object'),
        #     p = np.array(list(map(lambda x: x[1], move_p_as_list))))
        
        # Changed in favor of PDist
        # return random.choices(
        #     list(map(lambda x: x[0], move_p_as_list)),
        #     weights=list(map(lambda x: x[1], move_p_as_list))
        #     ).pop()
        return move_probabilities_as_PDist.sample()
        
    
    raise ValueError('Unknown return type ' + str(return_type))

def _tree_policy(core_tree, game_state_calculator, move_selector):
    address = []
    current = core_tree.root
    currently_at_root = True
    game_iterator = game_state_calculator.move_iterator()
    while True:
        if len(current.moves) == 0:
            # No valid moves; the game is over.
            address.append((current, None, None))
            hit_ghost = False
            break
        move = move_selector(current)
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


def _extension_policy(node, move, outcome,
                     prior_probabilities, new_players_turn):
    
    node.moves[move].outcome_dict[outcome] = CoreSearchNode(
        prior_probabilities, new_players_turn)


def _back_up_policy(address, evaluation):
    
    for node, game_move, outcome in reversed(address):
        if game_move is not None:
            move = node.moves[game_move]
            #Update the action value according to the player's 
            #expected gain determined by the lookahead.
            player = node.player
            move.action_value = (move.action_value*move.visits +
                        evaluation[player])/(move.visits + 1)
            move.visits = move.visits +1