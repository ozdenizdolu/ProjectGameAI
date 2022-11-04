# This module is responsible for the organization of game-independent
# information in the search tree.

# This module is freely used in the mcts package. Any change to this module
# should be applied carefully and, if possible, avoided.

import networkx as nx
from treelib import Tree
import itertools


class CoreSearchTree:
    
    def __init__(self, root_move_prior_probabilities, root_player):
        self.root = CoreSearchNode(root_move_prior_probabilities, root_player)
    
    #TODO send these visualizations to search tree.
    def visualize_nx(self):
        G = nx.Graph()
        G.add_node(self.root)
        current_nodes = [self.root]
        while len(current_nodes) > 0:
            node = current_nodes.pop()
            children = [child for move in node.moves.values() for child in move.outcome_dict.values()]
            G.add_edges_from(itertools.product([node],children))
            current_nodes.extend(children)
        nx.draw(G)
    
    def visualize_treelib(self):
        self.as_treelib_tree().show()
        
    def as_treelib_tree(self):
        tree = Tree()
        current_nodes = [(self.root, 0)]
        tree.create_node('Root', 0)
        i = 1
        while len(current_nodes) > 0:
            node, node_id = current_nodes.pop()
            for game_move, move in node.moves.items():
                for child in move.outcome_dict.values():
                    tree.create_node('{} --> {}'.format(str(game_move),str(move))
                                     , i, node_id)
                    current_nodes.append((child, i))
                    i += 1
        return tree

class CoreSearchNode:
    
    def __init__(self, prior_probabilities, player):
        # Player is needed for backup policy
        self.player = player
        self.moves = {move: CoreSearchMove(prior_probability) 
                for move, prior_probability in prior_probabilities.items()}    #EXPECT prior_probabilities to be a mapping supporting items(), EVALUATOR
        

class CoreSearchMove:

    def __init__(self, prior_probability):
        self.prior_probability = prior_probability
        self.visits = 0
        self.action_value = 0
        self.outcome_dict = {}
        
    def __str__(self):
        return ('v = ' + str(self.visits)
                + ' --- Q = ' + str(self.action_value)
                + ' --- p = ' + str(self.prior_probability))
        

