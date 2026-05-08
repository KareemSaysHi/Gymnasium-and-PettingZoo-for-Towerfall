from pettingzoo import ParallelEnv
from typing import Any, Mapping
from common.logging_options import default_logging
from towerfall import Towerfall
from agents import DummyAgent
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np

from gymnasium.utils import EzPickle # for some stupid bug


class TowerFallNoArrows(ParallelEnv, EzPickle):
    metadata = {
        "name": "TowerFallNoArrows",
    }

        
    def __init__(self, fps=10000, frames_per_action=5):
        #initialize towerfall environment
        self.towerfall = Towerfall( 
            verbose=1,
            config=dict(
            mode='sandbox',
            level='1',
            fps=fps,
            agentTimeout='00:00:10',
            solids=[[1] + [0] * 30 + [1]] * 14 + [[1] * 32] + [[1] + [0] * 30 + [1]] * 9,
            agents=[
                dict(type='remote', archer='green'),
                dict(type='remote', archer='blue'),
            ],
            )
        )
        
        #initialize required petting zoo variables:
        self.agents = ["archer_0", "archer_1"] 
        self.possible_agents = ["archer_0", "archer_1"] 
        self.observation_spaces = {a: Box(low=np.array([0, 0, -10, -5, 0, 0, -10, -5]), high=np.array([320, 240, 10, 5, 320, 240, 10, 5])) for a in self.agents} #me_px, me_py, me_vx, me_vy, and the others for other
        self.render_mode = "human"
        self.frames_per_action = frames_per_action #number of frames we wait per applying two actions.
        self._ezpickle_args = ""
        self._ezpickle_kwargs = {}
        self.action_spaces = {a: MultiDiscrete([5, 3]) for a in self.agents} #first is direction (none, up, down, left, right), second is buttons (none, jump, dash)  

        self.timestep = None
        self.max_steps = 600 #600 frames

        #initialize towerfall variables
        self.connections = {a: self.towerfall.join(timeout=10, verbose=1) for a in self.agents}

        self.agent_objects = {a: DummyAgent(self.connections[a], attack_archers=True) for a in self.agents}
    
        self.reset()


#----------------------- HELPER FUNCTIONS ----------------------

    # get observations of agents
    def get_observations(self):
        observations = []

        def build_obs(agent):
            ao = self.agent_objects[agent]
            observations = []
            observations.append(ao.pos['x'])
            observations.append(ao.pos['y'])
            observations.append(ao.vel['x'])
            observations.append(ao.vel['y'])
            return np.array(observations)

        return {
            "archer_0": np.concatenate([build_obs("archer_0"), build_obs("archer_1")]),
            "archer_1": np.concatenate([build_obs("archer_1"), build_obs("archer_0")])
        }
    
    #get rewards of agents
    # currently +1 if win, -1 if lose
    def get_rewards(self):
        whos_dead = []
        for a in self.agents:
            if self.agent_objects[a].dead == True:
                whos_dead.append(a)

        if len(whos_dead) != 1: #if not normal win condition
            return {
                "archer_0":  -0.1,
                "archer_1":  -0.1,
            }
        if whos_dead[0] == "archer_0":
            return {
                "archer_0":  -1.0,
                "archer_1":  1.0,
            }
        return {
            "archer_0":  1.0,
            "archer_1":  -1.0,
        }

    #how do we deal with rewards stuff:
    # I want to be able to celebrate with 
    
    # checks if terminated (terminates if someone is dead)
    def get_terminations(self):
        whos_dead = []
        for a in self.agents:
            if self.agent_objects[a].dead == True:
                whos_dead.append(a)
        if len(whos_dead) > 0:
            return {
                "archer_0":  True,
                "archer_1":  True
            }
        return {
            "archer_0":  False,
            "archer_1":  False
        }

    # checks if timeout
    def get_truncations(self): 
        time_up = self.timestep >= self.max_steps
        return {a: time_up for a in self.agents}
    
    # update agent data
    def get_agent_updates(self):
        #we need to go through init and scenario states before getting to update state.
        while True:
            for a in self.agents:
                connection = self.connections[a]
                ao = self.agent_objects[a]
                game_state = connection.read_json()

                if game_state['type'] != 'update':
                    ao.act(game_state) #agent does stuff on backend to get data during init and scenario states
                
                ao.update_data(game_state) #only does something if type is update

            if game_state['type'] == 'update':
                break

#---------------- REQUIRED PETTING ZOO STUFF -------------------

    def reset(self, seed = None, options = None):
        #tricky issue: towerfall.send_reset cannot be called UNTIL both agents reply with some action in the current episode.     

        # if this isn't the trivial first reset, we need to act to exit lockstep
        if self.agent_objects["archer_0"].state_update:
            self.agent_objects["archer_0"].pressed.clear()
            self.agent_objects["archer_1"].pressed.clear()
            self.agent_objects["archer_0"].send_actions()
            self.agent_objects["archer_1"].send_actions()

        #send data to reset
        self.towerfall.send_reset([
            dict(type='archer', pos=dict(x=80, y=110)),
            dict(type='archer', pos=dict(x=240, y=110)),
        ])

        #get new observations
        self.get_agent_updates()
        observations = self.get_observations()

        #reset timestep
        self.timestep = 0

        infos = {a: {} for a in self.agents}

        return observations, infos


    def step(self, actions):
        directions = {a: actions[a][0] for a in self.agents}
        buttons = {a: actions[a][1] for a in self.agents}

        dir_to_input = ['', 'l', 'r', 'u', 'd']
        but_to_input = ['', 'j', 'z']

        for i in range(self.frames_per_action):
            for a in self.agents:
                ao = self.agent_objects[a]
                if directions[a] > 0:
                    ao.press(dir_to_input[directions[a]])
                if buttons[a] > 0:
                    ao.press(but_to_input[buttons[a]])
                ao.send_actions()
            self.get_agent_updates()
              
        self.timestep += self.frames_per_action

        observations = self.get_observations()
        rewards = self.get_rewards()
        terminations = self.get_terminations()
        truncations = self.get_truncations()
        
        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]
            

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        #run the close thing
        self.towerfall.close_all()