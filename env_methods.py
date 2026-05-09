import numpy as np
from arrow_methods import Arrow

'''
Given an agent object, build the necessary observation dictionary needed for step() or model input
'''
def build_obs(agent_obj):
    observations = {}

    #build observations of both agents
    observations["my_movement"] = agent_obj.my_movement
    observations["my_arrows"] = agent_obj.my_arrows
    observations["your_movement"] = agent_obj.your_movement
    observations["your_arrows"] = agent_obj.your_arrows
    observations["my_facing"] = agent_obj.my_facing

    #build observations of arrows
    '''IMPORTANT: we control the order of the arrows in the observation slots, in the order of "importance" to the agent, where order is the dictionary ordering on
        - is traveling toward agent (i.e. positive dot product)
        - is no longer flying
        - how far is agent from arrow
    '''

    #1) get the list of arrows
    arrows = []
    for entity in agent_obj.state_update['entities']:
        if "arrowType" in entity:
            arrows.append(Arrow(entity))

    #2) sort the arrows
    sorted_arrows = sorted(arrows, key = lambda x: (not x.isDangerousTo(agent_obj), not x.is_grounded, x.distanceToAgent(agent_obj)))

    #3) build the arrow data in the observations:
    for i in range (len(sorted_arrows)):
        curr_arrow = sorted_arrows[i]
        observations[f"arrow{i}_exists"] = 1
        observations[f"arrow{i}_pos-vel"] = np.concatenate([curr_arrow.position, curr_arrow.velocity])            
        observations[f"arrow{i}_flying"] = int(not curr_arrow.is_grounded)
        observations[f"arrow{i}_grounded"] = int(curr_arrow.is_grounded)

    #4) pad the remaining slots
    for i in range (len(sorted_arrows), 6):
        observations[f"arrow{i}_exists"] = 0
        observations[f"arrow{i}_pos-vel"] = np.zeros(4, dtype=np.float32)            
        observations[f"arrow{i}_flying"] = 0
        observations[f"arrow{i}_grounded"] = 0

    return observations

def build_action_masks(agent_obj):
 
    direction_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    button_mask = [1, int(agent_obj.can_jump), 1, (agent_obj.my_arrows > 0)]
    #button order is nothing, jump, dash, shoot

    return np.array(direction_mask + button_mask)
