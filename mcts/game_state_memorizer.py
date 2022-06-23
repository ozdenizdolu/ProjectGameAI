# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 01:03:16 2022
"""

import random

# from numpy.random import default_rng


# rng = default_rng()


class GameStateMemorizer:
    '''
    This class is a concrete implementation of a GameStateCalculator. This one works by
    holding every visited game_state in memory.
    '''
    
    def __init__(self, root_game_state):
        self._root = MemorizerNode(root_game_state)
    
    def move_iterator(self):
        return MemorizedIterator(self._root)
    
    def initialize(self):
        return self._root.state, self._root._moves.keys() , self._root.turn
    

class MemorizedIterator:
    
    def __init__(self, root):
        self.current = root
        self._hit_ghost = False
        self._terminated = False
    
    def next_outcome(self, game_move):
        assert not self._terminated
        assert not self._hit_ghost, ('Failed use of Memorized iterator.' +
                                   'Check tree_policy for bugs. ' +
                                   'Iterator is called after it is finished.')
        move = self.current._moves[game_move]
        # selected_outcome = rng.choice(move.outcomes, p = move.p)
        selected_outcome = random.choices(move.outcomes, weights=move.p).pop()
        try:
            self.current = move.children_states[selected_outcome]
        except KeyError:
            self._hit_ghost = True
            self._leading_game_move = game_move
            self._leading_outcome = selected_outcome

        return selected_outcome
    
    def terminate(self):
        assert not self._terminated
        
        if self._hit_ghost:
            # This node was previously not visited. Recording it.
            new_node = MemorizerNode(self.current.state.after(
                self._leading_game_move, self._leading_outcome))
            self.current._moves[self._leading_game_move].children_states[self._leading_outcome] = new_node
            self.current = new_node
        else:
            # Only allow termination for visited nodes if they are 
            # game over nodes.
            assert self.current.is_game_over, (
                'Failed use of Memorized iterator.' +
                ' Check tree_policy for bugs. ' +
                'The iteration is finished on' +
                ' a non-leaf node.')
        
        #For assertion checks
        self._terminated = True
        
        return (self.current.is_game_over, self.current.game_final_evaluation,
                self.current.state, self.current._moves.keys(),
                self.current.turn)
        

class MemorizerNode:
    """
    ...
    
    Contains possible moves.
    """

    def __init__(self, game_state):
        # Request all the important knowledge from the game_state.
        self.state = game_state
        self.is_game_over = self.state.is_game_over()                         #EXPECT is_game_over() from GameState
        if self.is_game_over:
            self.game_final_evaluation = self.state.game_final_evaluation()                   #EXPECT game_final_evaluation from GameState NOTE!: it should be a mapping form players to numbers.
        else:
            self.game_final_evaluation = None
        self._moves = {}
        for move in game_state.moves():                                                     #EXPECT moves() from GameState
            self._moves[move] = MemorizerMove(*game_state.outcomes(move))                 #EXPECT a tuple of lists of same size.
        self.turn = self.state.turn()

class MemorizerMove:
    
    def __init__(self, outcomes, p):
        self.outcomes = outcomes
        self.p = p
        self.children_states = {}
        



