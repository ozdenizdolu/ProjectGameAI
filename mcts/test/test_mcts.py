import unittest

import numpy as np

from evaluator.evaluators import random_playout_evaluator
from .. import mcts, uct


class TestsUCT(unittest.TestCase):
    
    @classmethod
    def mock_game_generator(cls, children):
        # Create a gamestate class of a game of depth 1
        # and return an instance of it
        # for testing purposes. children is a dict whose
        # keys are supposed moves and their outcomes are
        # the result of the game.
        
        class MockGameState:
            
            def __init__(self, is_leaf, identity):
                self.is_leaf = is_leaf
                self.identity = identity
            
            def turn(self):
                return 0
            
            def game(self):
                return None
            
            def is_game_over(self):
                return self.is_leaf
            
            def game_final_evaluation(self):
                return {0: children[self.identity]}
            
            def moves(self):
                if self.is_leaf:
                    return []
                else:
                    return children.keys()
                
            def outcomes(self, move, pick_one = False):
                if not pick_one:
                    return ['OUTCOME'], [1]
                else:
                    return 'OUTCOME'
            
            def after(self, move, outcome):
                if move in children.keys() and outcome in ['OUTCOME', None]:
                    return MockGameState(True, move)
                else:
                    raise ValueError()
                
        return MockGameState(False, None)
    
    def test_balanced(self):
        """Test whether the tree is balanced when all moves
        are equally very bad."""
        
        def with_parameters(times, branching, evaluation, correct_visits):
            
            tree = uct(self.mock_game_generator(
                {i: evaluation for i in range(branching)}),
                times, 1, return_type = 'tree')
            
            leafs = tree._core_tree.root.moves.values()
            
            self.assertEqual(len(leafs), branching)
            # print(set(map(lambda child: child.visits, leafs)))
            self.assertEqual(correct_visits,
                             len(set(map(lambda child: child.visits, leafs))))
        
        # 4212 times expansion will lead to 2 nodes being expanded
        # one more times than the others. That would mean
        # there are 2 distinct number of visits.
        with_parameters(4212, 10, -100, 2)
        
        # 10 divides 100 so every node should be visited equally
        with_parameters(100, 10, -100, 1)
        
        # All will be visited the same number of times. They will all have
        #the same action value.
        with_parameters(1000, 10, 100, 1)
        
        # Every node will be expanded once. There is no time for exploitation.
        with_parameters(100, 10, 100, 1)
        
        # with_parameters(100, 10, 0, 10)
            
            
            
            
            