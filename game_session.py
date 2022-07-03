"""
---- Development Note ----
GameSessionTemplate need not be public. It is there to abstract the common
logic of game sessions. Each game session can (and should)
create their own documentation for the run method. Run method can return
things.
"""

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