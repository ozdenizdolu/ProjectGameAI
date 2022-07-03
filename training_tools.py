"""

"""

import math
import random
from collections import deque

import torch

from game_session import GameSessionTemplate
from mcts import mcts
import neural_network



# Let us not include Bayesian optimization for now in this method.
# Let us create something that works first.
def train(game,
          neural_network,
          mcts_steps_per_move,
          move_selector,
          pool_size,
          mini_batch_size,
          num_of_batches,
          pool_refresh_fraction,
          num_of_training_steps,
          learning_rate,
          temperature,
          exploration_constant,
          device,
          move_loss_function = None,
          eval_loss_function = None
          ):
    
    if move_loss_function is None:
        move_loss_function = torch.nn.CrossEntropyLoss()
    if eval_loss_function is None:
        eval_loss_function = torch.nn.MSELoss()
    
    #Initializing the training data pool
        
    training_pool = deque(maxlen=pool_size)
    
    #First, filling in the pool
    
    update_pool(training_pool, pool_size, game, neural_network,
                mcts_steps_per_move, exploration_constant,
                move_selector, temperature)
        
    print("Pool initialization is comlete!")
    
    #Start the main training loop
    
    optimizer = torch.optim.SGD(neural_network.parameters(), lr=learning_rate)
    
    for iteration in range(num_of_training_steps):
        
        # Training step
        # Telling PyTorch that we are training
        neural_network.train()
        
        for _ in range(num_of_batches):
            games = random.choices(list(training_pool), k = mini_batch_size)
            batch = [random.choice(game) for game in games]
            
            raw_in = list(map(lambda x: x[0], batch))
            raw_out = list(map(lambda x: x[1], batch))
            
            raw_moves = list(map(lambda x: x[0], raw_out))
            raw_evaluation = list(map(lambda x: x[1], raw_out))
            
            in_ = torch.cat([single.unsqueeze(0) for single in raw_in], dim=0).to(device)
            
            move_target = torch.cat([single.unsqueeze(0)
                                  for single in raw_moves], dim=0).to(device)
            evaluation_target = torch.cat([single.unsqueeze(0)
                                        for single in raw_evaluation], dim=0).to(device)
            
            move_tensor, evaluation_tensor = neural_network(in_)
            
            loss1 = move_loss_function(move_tensor, move_target)
            loss2 = eval_loss_function(evaluation_tensor, evaluation_target)
            
            total_loss = loss1 + loss2
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print("Mini batch complete. Loss is: {}".format(total_loss.item()))
            
        neural_network.eval()
        
        # Update the pool
        update_pool(training_pool, math.floor(pool_size*pool_refresh_fraction),
                    game, neural_network, mcts_steps_per_move,
                    exploration_constant, move_selector, temperature)
    
    #The training is complete.
    

    
def update_pool(pool, game_count, game, neural_network, mcts_steps_per_move,
                exploration_constant, move_selector, temperature):
    
    for _ in range(game_count):
        game_data = []
        TrainingGameSession(
            game,
            neural_network.as_evaluator(),
            game_data,
            mcts_steps_per_move,
            exploration_constant,
            move_selector,
            temperature).run()
        
        game_data = [
            (neural_network.state_to_input(state),
             neural_network.translate_out(move_distribution, evaluation))
            for state, move_distribution, evaluation in game_data]
        
        pool.append(game_data)


class TournamentGameSession(GameSessionTemplate):
    
    def __init__(self, game, agent_dict):
        self._game = game
        self._agent_dict = agent_dict
    
    def initialize(self):
        self._state = self._game.initial_state()
    
    def current_state(self):
        return self._state

    def proceed(self, player):
        move = self._agent_dict[player].ask(self._state)
        if move not in self._state.moves():
            raise ValueError('Player {} attempted an illegal move...'.format(
                player))
        self._state = self._state.after(move, None)
    
    def return_value(self):
        return self._state.game_final_evaluation()


class TrainingGameSession(GameSessionTemplate):
    
    def __init__(self,
                 initial_state,
                 evaluator,
                 output,
                 mcts_steps_per_move,
                 exploration_constant,
                 move_selector,
                 temperature):
        """
        Creates a new instance of a game session which records training
        data to @output.
        
        Parameters
        ----------
        output: an object supporting extend method
            The collection to which the training data of the game session will
            be appended to. The training data consists of tuples
            where the first element is the state of the game;
            the second element is the move probabilities generated by
            the mcts search, and the third element is the result of the
            game in dict format. 
        """
        self._evaluator = evaluator
        self._mcts_steps_per_move = mcts_steps_per_move
        self._exploration_constant = exploration_constant
        self._move_selector = move_selector
        self._temperature = temperature
        
        self._output = output
        self._temporary_distribution_list = []
        
        self._states = [initial_state]
        
    def initialize(self):
        pass
    
    def proceed(self, player):
        # The next move is determined by mcts. This method needs to
        # keep track of mcts probabilities of the search to create
        # the required training data.
        move_probabilities = mcts(
            self._states[-1],
            # Do not worry about the colours. Evaluators are colour-free.
            self._evaluator,
            self._mcts_steps_per_move,
            exploration_constant = self._exploration_constant,
            move_selector = self._move_selector,
            temperature = self._temperature,
            return_type = 'distribution')
        as_list = list(move_probabilities.items())
        selected_move = random.choices(
            [item[0] for item in as_list],
            weights = [item[1] for item in as_list])[0]
        
        self._temporary_distribution_list.append(move_probabilities)        
        self._states.append(self._states[-1].after(selected_move, None))

    def current_state(self):
        return self._states[-1]
    
    def finalize(self):
        #Record the result and output to the output list.
        result = self._states[-1].game_final_evaluation()
        self._output.extend([(state, distribution, result)
            for state, distribution
            in zip(self._states, self._temporary_distribution_list)])
        
    
    
