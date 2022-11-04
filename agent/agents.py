import random
import math

from mcts import mcts, uct
from miscellaneous import after, PDist

class RandomAgent:
    """
    The class of agents which play completely randomly.
    """
    
    def ask(self, state):
        return random.choice(state.moves())
    
    def __str__(self):
        return 'Random Agent'


class UCTAgent:
    """
    The class of agents which play using the UCT algorithm.
    """
    
    def __init__(self, times_per_move, 
                 exploration_constant=1, temperature=0.05):
        
        self._times_per_move = times_per_move
        self._exploration_constant = exploration_constant
        self._temperature = temperature
    
    def ask(self, state):
        return uct(
            state,
            self._times_per_move,
            self._exploration_constant,
            temperature = self._temperature,
            return_type = 'move')
    
    def __str__(self):
        return 'UCT agent'

class EvaluatorAgent:
    """
    Agent using an evaluator's probability distribution
    on moves and evaluation to decide on the move.
    """
    
    def __init__(self, evaluator,
                 distribution_weight = 1., evaluation_weight = 0.0):
        self._evaluator = evaluator
        self._distribution_weight = distribution_weight
        self._evaluation_weight = evaluation_weight
        self._does_consider_evaluation = evaluation_weight != 0.0
        if not math.isclose(distribution_weight + evaluation_weight, 1.0):
            raise ValueError('Sum of weights should be 1.')
        
    def ask(self, state):
        distribution_part = self._evaluator(state, state.moves(),
                                            state.turn())[0]
        
        if self._does_consider_evaluation:
            evaluation_part = {}
            for move in state.moves():
                selected_state = state.after(move, None)
                if selected_state.is_game_over():
                    raw_evaluation = state.game_final_evaluation()[state.turn()]
                else:
                    raw_evaluation = (self._evaluator(selected_state,
                                               selected_state.moves(),
                                               selected_state.turn())[1][
                                                   state.turn()])
                evaluation_part[move] = 10**-8 + (raw_evaluation + 1.) / 2.
                if evaluation_part[move] < 0 or evaluation_part[move] > 1+10**-7:
                    raise RuntimeError('''{} agent does not provide evaluation
                                       between -1 and 1'''.format(self))
            evaluation_part = PDist(*zip(*evaluation_part.items()),
                                    normalize=True,
                                    supports_hash=True)
            
            probabilities = {}
            for move in evaluation_part:
                probabilities[move] = (
                distribution_part[move] * self._distribution_weight
                + evaluation_part[move] * self._evaluation_weight)
        else:
            probabilities = distribution_part
        
        return random.choices(*zip(*probabilities.items()), k = 1)[0]
    
    def __str__(self):
        return 'Evaluator agent'

# TODO: redo this, mcts package has changed
# class MCTSAgent:
#     """
#     An agent which uses mcts search and an evaluator to decide their moves.
#     """
    
#     def __init__(self, evaluator, mcts_steps, move_selector,
#                  exploration_constant, temperature):
#         self._evaluator = evaluator
#         self._mcts_steps = mcts_steps
#         self._move_selector = move_selector
#         self._exploration_constant = exploration_constant
#         self._temperature = temperature
    
#     def ask(self, state):
#         return mcts.mcts(state, self._evaluator, self._mcts_steps,
#                          self._move_selector, self._exploration_constant,
#                          self._temperature, return_type='move')

#     def __str__(self):
#         return 'MCTS agent'




class UserConsoleAgentForTCT:
    
    def __init__(self):
        pass
    
    def ask(self, state):
        # Notify the spectator
        print('\n'+str(state))
        # Request move from the player
        print('\nYour turn...')
        while True:
            in_ = input()
            match = re.match('\d\d', in_)
            if match is None:
                if in_ == 'e':
                    return
                if in_ == 'p':
                    break
                print('Prompt not recognized.')
                continue
            x,y = map(int, match[0])
            x,y = x-1,y-1
            if (x,y) in map(lambda x: x[:-1],state.moves()):
                break
            else:
                print('Move is not possible.')
                continue
        
        return (x, y, state.turn())








