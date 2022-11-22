import random
import itertools

from mcts import mcts

# This do not reuse the search tree from the previous move because it is
# biased.
def generate_data(game,
                  termination_condition,
                  termination_number,
                  evaluator,
                  mcts_steps_per_move,
                  move_selector,
                  temperature,
                  dirichlet_noise_parameter = None,
                  noise_contribution = None,
                  shuffle=True,
                  verbose = True):
    """
    Generates a data using the evaluator with mcts.
    
    Uses mcts with evaluator to generate games of self-play. Those plays
    are used to generate output data. Data is an iterable consisting of 
    tuples of the form:
        (game_state, dist_on_moves, game_evaluation)
    where game_state is a game state encountered during the self-play,
    dist_on_moves is a dict with keys as game moves and values as the
    probabilities given by the mcts search during the game. game_evaluation
    is the result of that self-play game.
    """
    
    data = []
    for game_number in itertools.count(1):
        raw_game_data = []
        # play the game.
        game_state = game.initial_state()
        # Do-while
        while True:
            # Run mcts
            move_dist = mcts(game_state,
                        evaluator,
                        mcts_steps_per_move,
                        move_selector=move_selector,
                        temperature=temperature,
                        return_type = 'distribution',
                        dirichlet_noise_parameter=dirichlet_noise_parameter,
                        noise_contribution=noise_contribution)
            # Record the distribution
            raw_game_data.append((game_state, move_dist))
            # Pick a move according to the distribution.
            move = move_dist.sample()
            # Advance the game with normal rules
            game_state = game_state.after(move, None)
            if game_state.is_game_over():
                break
        
        result = game_state.game_final_evaluation()
        
        data.extend([(state, move_dist, result)
                     for state,move_dist in raw_game_data])
        
        if verbose:
            print("Game "+str(game_number) +" is over.")
            print('{} {}/{}'.format(
                termination_condition,
                game_number if termination_condition == 'g'
                else len(data),
                termination_number))
            
        # Check termination condition
        if termination_condition == 'g':
            if game_number >= termination_number:
                break
        elif termination_condition == 's':
            if len(data) >= termination_number:
                break
        else:
            raise ValueError()
    
    if shuffle:
        random.shuffle(data)
    return data
