# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 22:34:25 2022
"""

class SearchTree:
    
    def __init__(self, core_tree, game_state_calculator):
        self._core_tree = core_tree
        self._game_state_calculator = game_state_calculator
        
    def _components(self):
        return (self._core_tree, self._game_state_calculator)
    
    #TODO improve this view to be more user friendly. Document.
    def root(self):
        "Use this to get a view on the tree."
        return self._core_tree.root
    
        