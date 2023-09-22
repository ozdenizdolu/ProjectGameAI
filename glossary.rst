agent
-----

An interface capable of playing games. An agent implements
ask(game state) funtion which returns the move selected by the agent.
Library provides some agents in agent package.


evaluation
----------

The evaluation of a game state. Evaluation is a mapping of players
to payoffs.


evaluator
---------

A callable which is used by Monte Carlo tree search to guide the search.
An evaluator provides probability distributions on possible moves and the
evaluation of game states. It accepts three parameters: game_state, moves, 
player and outputs a tuple (prior_probabilities, evaluation) where prior 
probabilities is a mapping of legal moves to probabilites which encodes
a probability distribution on legal moves at game_state, and evaluation
is a mapping of players of the game to the
evaluation of game_state with respect to the respective players.
The inputs, moves and player, are the legal moves at game_state
and the player having the turn, respectively; they can also be deduced from
the game state, but they are provided to for performance.


game
----

An object identifying the game. These objects can offer auxiliary 
algorithms which can increase the performance of the MCTS search by
, for example, providing a game-specific game state calculator. A game
can support the following function:

def game_state_calculator(game_state)
"""Returns a game state calculator object initialized at game_state game state.

See mcts.game_state_calculators.__init__ for details."""


game session
------------

This is used to simulate games. See the module with the same name
for the interface and an implementation. Currently only ConsoleGameSession
is provided by the library, which can be used to play/observe a game
between agents in the console.


game state
----------

Immutable and hashable object representing a definite state of the game at
a point. The game states need not provide how the game ended up there; it should only
contain all the information needed for the simulation of the game from that
point on. In chess, for example, this requires the knowledge of
the board positions encountered before reaching the current point because
threefold repetition rule needs to be resolved for forward simulation. 
Game states are expected to be fast and memory efficient objects, and
they are thought of as data, storing only the variables needed
for the generation of the possible game states for one move ahead.

The methods expected from a game state are:

game()
"""Returns the game object this game state belongs to."""


is_game_over()
"""Returns the information as a Boolean variable."""


turn()
"""Return the player who is going to play at the moment."""


moves()
"""Returns the available moves as an iterable. Moves should be hashable.

The exact nature of the moves is not important, but each move should be distinguishable
and the returned moves should be compatible to use with the functions, outcomes and after."""


outcomes(move, pick_one=False)
"""Returns the probabilistic outcomes after the given move at the current state of the
game together with their probabilities.

This functionality is for probabilistic games, if the game is deterministic, they
should return a single outcome with probability 1. For backgammon, this should
return the oppenent's possible dice rolls with their corresponding probabilities. 
The returned value is a tuple (outcomes, p) where each is an iterable of same length,
p denoting the probabilities of the corresponding outcome. Sum of p's should be 1. 
Outcomes can be anything, but they must be hashable.

If pick_one = True then randomly selects one of the outcomes according to p."""


game_final_evaluation()
"""Returns a mapping where keys are players and the values are the points they get.

If is_game_over() is not True, then the behaviour is not specified. For win-lose
games, mapping winning player to 1 and losing player to -1 is a good choice. The
magnitute of the points will affect the search algorithms."""


after(move, outcome)
"""Returns the game state obtained by making the move and observing the outcome.

If outcome is None, then picks one randomly according to the rules of the game.
In that case should be equivalent to after(move, state.outcomes(move, pick_one=True))."""



game state calculator
---------------------
This is a dependency injection interface for mcts. Some games may have
game states which are memory intensive to hold in memory. The structure 
of the mcts search may be used to optimize this. Not necessary for many games. 

The library provides a default one, which memorizes every game state
encountered during the search. To provide one for a game, read the game
item in the glossary and the __init__ file in mcts.game_state_calculators.


player
------
immutable constants representing the players of the game.