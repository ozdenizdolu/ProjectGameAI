agent
-----

An object capable of playing games. An agent implements
ask(game state) funtion which returns the move selected by the agent.

evaluation
----------

The evaluation of a game state. Evaluation is a mapping of players
to payoffs.



evaluator
---------

A function which is used by Monte Carlo tree search to provide probability
distributions on possible moves and the evaluation of game states. It
accepts three parameters: game_state, moves, player and outputs a tuple
(prior_probabilities, evaluation) where prior probabilities is a mapping
of legal moves to probabilites which encodes a distribution on legal moves
at game_state, and evaluation is a mapping of players of the game to the
evaluation of game_state with respect to the respective players.
The inputs, moves and player, are the legal moves at the game_state
and the player having the turn, respectively; they are provided only for
performance.


game
----

game session
------------



game state
----------

Immutable and hashable object representing a definite state of the game at
a point.
The game states need not know how the game ended up there; it should only
contain all the information needed for the simulation of the game from that
point on. In chess, for example, this requires the knowledge of
the board positions encountered before reaching the current point because
threefold repetition rule needs to be resolved for forward simulation. 
Game states are expected to be fast and memory efficient objects, and
they are thought of as data storing only the variables needed
for the generation of the possible game states one move later.
The methods expected from a game state are:

# is_game_over()
# outcomes(move, pickone=False)
# game_final_evaluation()
# game()
# turn()
# moves()
# after(move, outcome)



game state calculator
---------------------


player
------
immutable constants representing the players of the game.