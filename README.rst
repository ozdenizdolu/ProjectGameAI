An Implementation of AlphaZero
==============================

This project is created by a machine learning group at Boğaziçi University.
The aim is to implement Monte Carlo tree search algorithm, upper confidence
bounds applied to trees algorithm[2]_, and AlphaZero [1]_ algorithm to understand
them better. AlphaZero is an algorithm developed by Deepmind which learns to play games
at superhuman level starting with no knowledge of the game except the rules. 

Usage
-----

mcts folder contains the Monte Carlo tree search algorithm, and the special
case of it called UCT, upper confidence bounds applied to trees [2]_. To
use these algorithms use 

import ProjectGameAI as ai

# use ai.mcts(), ai.uct() or other functionalities. See init file for examples.

See the glossary for the terms used. Some other functionalities can be used using
the glossary as a guide. The rest is under development; for those see the branch
development.


Structure of the Project
------------------------

See glossary file for the meaning of the terms used in the project. The
projects consists of several modules. ProjectGameAI/mcts provides
the mcts search functionality for any evaluator, which need not be a neural
network. It provides special cases of mcts search like UCT algorithm as well.
ProjectGameAI/game provides core functionality for supported games like
reversi and tictactoe. Adding new games will be compatible with the rest
of the project provided that the game and game states conform to the
interface explained in glossary. Probabilistic games are also supported.
We are currently experimenting with reinforcement learning, but it is under
development. The project also contains implementation of several
types of neural networks including transformer, but they are not documented,
and are in the development branch.

Contribute
----------

We welcome contributions to this project! If you would like to contribute, 
contact group's founder Fırat Kıyak through firat.kiyak@boun.edu.tr. You can
also make a pull request for good first issues.

Authors
-------

- Fırat Kıyak
- Özdeniz Dolu


References
----------

.. [1] Silver, David, et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science 362.6419 (2018): 1140-1144.
.. [2] Kocsis, Levente, and Csaba Szepesvári. "Bandit based monte-carlo planning." European conference on machine learning. Berlin, Heidelberg: Springer Berlin Heidelberg, 2006.