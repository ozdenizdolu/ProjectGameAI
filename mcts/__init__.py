"""
This package provides Monte Carlo tree search algorithm.

Use 

from mcts import mcts, uct

for importing the functions. Use

from mcts.move_selectors import ...

for importing special default move selectors to use in mcts.
"""

from ._mcts_logic import mcts
from ._uct import uct