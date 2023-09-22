# This module contains the standard uct algorithm

from ..evaluator.evaluators import random_playout_evaluator
from . import mcts
from .move_selectors import UCT_move_selector_factory

def uct(data,
        times,
        exploration_constant,
        temperature = 1,
        return_type = 'move'):
    """Provides the upper confidence bounds applied to trees algorithm.
    
    This is a special case of Monte Carlo tree search. This algorithm
    was introduced in the following article:
        
        Kocsis, L. and Szepesvári, C., 2006, September.
        Bandit based monte-carlo planning.
        In European conference on machine learning (pp. 282-293).
        Springer, Berlin, Heidelberg.
    """
    return mcts(
        data,
        random_playout_evaluator,
        times,
        move_selector = UCT_move_selector_factory(exploration_constant),
        temperature = temperature,
        return_type = return_type)