from pettingzoo import ParallelEnv
from towerfall import Towerfall
from agents import PPOAgentShooting_V0
from gymnasium.spaces import Box, MultiDiscrete, Dict, Discrete
import numpy as np

import env_methods as env_methods 

''' TowerFallEnv is a PettingZoo Environment for interacting with TowerFall '''
class TowerFallEnv(ParallelEnv):
    metadata = {
        "name": "TowerFallEnv",
    }
        
    def __init__(self, fps=10000, frames_per_action=5):        

        self.towerfall = False

        self.has_connected = False
        
        #initialize required petting zoo variables:
        self.agents = ["archer_0", "archer_1"] 
        self.possible_agents = ["archer_0", "archer_1"] 
        self.render_mode = None 
        self.timestep = None
        self.fps = fps
        self.frames_per_action = frames_per_action #number of frames we wait per applying two actions.
        self.max_steps = 600 #600 frames, not 600 actions


        ''' The observation space:
        for players, we keep track of
            position, velocity, direction facing, arrow count
        for each arrow, we keep track of
            position, velocity, flying, grounded
        '''

        obs_dict = {
            "my_movement": Box(low=np.array([0, 0, -10, -5]), high=np.array([320, 240, 10, 5])), #position and velocity
            "my_arrows": Discrete(9), #0 through 6
            "your_movement": Box(low=np.array([0, 0, -10, -5]), high=np.array([320, 240, 10, 5])), #position and velocity
            "your_arrows": Discrete(9), #0 through 6
            "my_facing": Discrete(2) #either 0 or 1, for facing left, facing right
        }

        for i in range (6):
            obs_dict[f"arrow{i}_exists"] = Discrete(2) #is the arrow slot filled
            obs_dict[f"arrow{i}_pos-vel"] = Box(low=np.array([0, 0, -10, -10]), high=np.array([320, 240, 10, 10])) #position and velocity
            obs_dict[f"arrow{i}_flying"] = Discrete(2) #flying
            obs_dict[f"arrow{i}_grounded"] = Discrete(2) #grounded

        self.observation_spaces = {a: Dict(obs_dict) for a in self.agents} #make obs_dict into PettingZoo Space

        self.action_spaces = {a: MultiDiscrete([9, 4]) for a in self.agents} #first is direction (none, and then the 8 directions), second is buttons (none, jump, dash, shoot)  

        #important note: in Towerfall, typically the shoot button fires on release.  In our implementation, every press of a shoot button will fire an arrow


#----------------------- HELPER FUNCTIONS ----------------------

    # get observations of agents
    def get_observations(self):
        return {
            a: env_methods.build_obs(self.agent_objects[a]) for a in self.agents
        }
    
    # get action masks
    def action_masks(self):
        #print(f"action mask: {env_methods.build_action_masks(self.agent_objects['archer_0'])}")
        #print(f"can jump: {self.agent_objects['archer_0'].can_jump}")
        #print(f"can dodge: {self.agent_objects['archer_0'].can_dodge}")
        #print(f"dodge_cooldown: {self.agent_objects['archer_0'].dodge_cooldown}")
        return {
            a: env_methods.build_action_masks(self.agent_objects[a]) for a in self.agents
        }
    
    #get rewards of agents
    def get_rewards(self):
        whos_dead = []
        for a in self.agents:
            if self.agent_objects[a].dead == True:
                whos_dead.append(a)

        if len(whos_dead) != 1: #if both are still alive
            return {
                "archer_0":  -0.003, #small negative reward to discourage camping
                "archer_1":  -0.003,
            }
        
        if whos_dead[0] == "archer_0": # +1/-1 for winning/losing
            return {
                "archer_0":  -1.0,
                "archer_1":  1.0,
            }
        return {
            "archer_0":  1.0,
            "archer_1":  -1.0,
        }
    
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
        if self.has_connected == False:
            
             #Initialize Towerfall Environment
            self.towerfall = Towerfall( 
                verbose=1,
                config=dict(
                mode='sandbox',
                level='1',
                fps=self.fps,
                agentTimeout='00:00:10',
                solids =  [[1] + [0] * 30 + [1]] * 14 + [[1] * 32] + [[1] + [0] * 30 + [1]] * 9, #general flat terrain
                agents=[
                    dict(type='remote', archer='green'),
                    dict(type='remote', archer='blue'),
                ])
            )

            #initializing towerfall variables
            self.connections = {a: self.towerfall.join(timeout=10, verbose=1) for a in self.agents}
            
            self.agent_objects = {a: PPOAgentShooting_V0(
                connection = self.connections[a], 
                mode = "training",
                frames_per_action= self.frames_per_action)
                for a in self.agents}

            self.has_connected = True



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

        for a in self.agents:
            self.agent_objects[a].reset_vars()

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

        any_agents_dead = False
        for i in range(self.frames_per_action):
            for a in self.agents:
                if self.agent_objects[a].dead:
                    any_agents_dead = True

            if any_agents_dead:
                break

            for a in self.agents:
                ao = self.agent_objects[a]
                remove_shoot = True if i == self.frames_per_action - 1 else False
                ao.send_env_actions(directions[a], buttons[a], remove_shoot)
            self.get_agent_updates()
              
        self.timestep += self.frames_per_action

        observations = self.get_observations()
        rewards = self.get_rewards()
        terminations = self.get_terminations()
        truncations = self.get_truncations()
        
        infos = {a: {"dead": self.agent_objects[a].dead} for a in self.agents}

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
