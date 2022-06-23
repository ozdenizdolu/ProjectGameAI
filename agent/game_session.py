# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:15:24 2022
"""
import itertools


class Player:
    """An agent capable of playing a game.
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    
class RandomPlayer:
    
    # def __init__(self, game):
    #     self._game = game
        
    def request_move(self, game_state):
        return random.choice(list(game_state.moves()))
        


def GameSession:
    
    def __init__(self, game, players, spectators):
        if set(players.keys()) != set(game.players()):
            raise ValueError('Players do not match the characters'
                             + 'in the game.')
        self.game = game
        self.players = players
        self.spectators = spectators
        
    #TODO
    def start(self):
        
        # self.state_history = [self.game.initial_state()]
        # self.move_history = []
        # self.outcome_history = []
        # self.turn_history = []
        
        #for chart in self.charts: chart.keep(self) maybe?
        
        # for spectator in self.spectators:
        #     # Notify everyone that the game has started
        #     self.spectator.notify_start(self)
        
        self.initiate_game_start_event()
        
        while not self.is_game_over:
            
            #maybe create Move Played event?
            #self.initiate_move_played_event()
            # player = players[state_history[-1].turn()]
            # move = player.request_move(state_history[-1])
            # outcome = state_history[-1].outcomes(move, pick_one=True)
            # state_history.append(state_history[-1].after(move, outcome))
            
            # for agent in itertools.chain(players.values(), spectators):
            #     agent.notify(player, *state_history[-2:], move, outcome)
            
            self.initiate_move_event()
            
        
        self.initiate_game_over_event()
            
        # for agent in itertools.chain(players.values(), spectators):
            # Notify everyone that the game has ended
            # agent.notify_end(state_history)
            
            
            #initiate game_has_ended event instead.
        

class Event:
    """"""
    
    
    

def session(game, players, spectators):
    """
    Start a game between @players.
    
    Sends game states to the players
    when it is their turn, and expects a move. After each
    move played, it notifies spectators regarding the current
    state of the game.
    
    Parameters
    ----------
    game: Game object
        The game to be played. See game module for supported games.
    
    players: mapping object
        The keys are the players of the @game. The corresponding
        values are the player agents who assume the role of
        that player.
        
        For example, in reversi, the keys are game.reversi.WHITE
        and game.reversi.BLACK.
    
    spectators: a list of spectator objects
        These objects will be notified about the updates about
        the game.
    """
    
    # gs = GameSession(game, players, spectators)
    
    if set(players.keys()) != set(game.players()):
        raise ValueError('Players do not match the characters'
                          + 'in the game.')
    
    # class Information:
    #     pass
    
    # information = Information()
    
    state_history = [game.initial_state()]
    
    for spectator in spectators:
        # Notify everyone that the game has started
        spectator.notify_start(state_history[-1])
    
    while not game_state.is_game_over():
        player = players[state_history[-1].turn()]
        move = player.request_move(state_history[-1])
        outcome = state_history[-1].outcomes(move, pick_one=True)
        state_history.append(state_history[-1].after(move, outcome))
        
        for agent in itertools.chain(players.values(), spectators):
            agent.notify(player, *state_history[-2:], move, outcome)
        
    for agent in itertools.chain(players.values(), spectators):
        # Notify everyone that the game has ended
        agent.notify_end(state_history)










