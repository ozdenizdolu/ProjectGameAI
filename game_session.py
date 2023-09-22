import time
from abc import ABC, abstractmethod

class GameSessionTemplate(ABC):
    """
    Template method design pattern applied for game sessions.
    
    run method is the template for the core logic of a game session. Extend
    this class by overriding other methods for specialized game sessions.
    """
    
    def run(self):
        """
        The template method for game play.
        
        This method makes the game going until the game is over by using
        the functionalities implemented by a concrete instance. The concrete
        implementations may return something other than None.
        
        Warnings
        --------
        This method is not intended to be extended. 

        """
        self.initialize()
        while True:
            current = self.current_state()
            if current.is_game_over():
                break
            self.proceed(current.turn())
        self.finalize()
        
        return self.return_value()

    def initialize(self):
        """
        Initializes the game.
        """

    def finalize(self):
        """
        Finalizes the game.
        """
    
    @abstractmethod
    def current_state(self):
        """Returns the current state of the game."""
        
    
    @abstractmethod
    def proceed(self, player):
        """
        The main logic of the session. Ask the next player for their move,
        and make the game proceed by a single step.

        """
    def return_value(self):
        """
        If concrete implementations wishes run to return a value, then they
        can return it with this function.
        """
        

class ConsoleGameSession(GameSessionTemplate):
    """Make the agents play the game and observe it from the console."""
    
    def __init__(self, state, agent_dict, * , delay = 0.6):
        self._state = state
        self._agent_dict = agent_dict
        self._delay = delay
        self._history = [self._state]

    def current_state(self):
        return self._state
    
    def proceed(self, player):
        time.sleep(self._delay)
        agent = self._agent_dict[player]
        print('Agent {} as {} is thinking...\n'.format(
            str(agent), str(player)))
        move = agent.ask(self.current_state())
        if move not in self.current_state().moves():
            raise RuntimeError('Agent {} committed an illegal move...'.format(
                agent))
        self._state = self._state.after(move, None)
        print('Agent {} has played {}\n'.format(str(agent), str(move)))
        print('Current Game State:\n{}'.format(str(self._state)))
        self._history.append(self._state)
    
    def initialize(self):
        print('The game is starting...\n')
        print('The agents play as:')
        for player, agent in self._agent_dict.items():
            print('{} plays as {}'.format(str(agent), str(player)))
    
    def finalize(self):
        print('The game has ended.\n{}\n'.format(str(self._state)))
        print('''The results are: 
              \n{}'''.format(str(self._state.game_final_evaluation())))
        print('The agents play as:')
        for player, agent in self._agent_dict.items():
            print('{} plays as {}'.format(str(agent), str(player)))
    
    def return_value(self):
        return self._history
    