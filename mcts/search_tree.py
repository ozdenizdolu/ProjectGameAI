class SearchTree:
    """A search tree is a data structure on which MCTS algorithm can
    be run.
    
    MCTS algorithm can be run several times on a search tree. Each MCTS
    call will change the search tree and the total change will be the same
    as what would happen if all calls were made as a single call. It is up to
    the caller to check whether those calls were made with the same
    parameters; it is possible to use different search parameters on
    the same tree. 
    """
    
    def __init__(self, core_tree, game_state_calculator):
        self._core_tree = core_tree
        self._game_state_calculator = game_state_calculator
        
    def _components(self):
        return (self._core_tree, self._game_state_calculator)
    
    # TODO improve this view to be more user friendly. Document.
    # Use a specialized object to provide the view. Do not give the
    # original tree node for visualization.
    
    # TODO: Also make sure that this function is not the same as the
    # one used in the logic of mcts package.
    # def root(self):
    #     "Use this to get a view on the tree."
    #     return self._core_tree.root