import numpy as np

''' An Arrow class stores relevant information about arrows, and helper functions that help us sort the importance of curves later '''
class Arrow():
    def __init__(self, entity = None):
        self.dict = entity
        self.position = np.array([entity['pos']['x'], entity['pos']['y']])
        self.velocity = np.array([entity['vel']['x'], entity['vel']['y']])
        self.is_grounded = not entity['canHurt']

    def isDangerousTo(self, agent_obj):
        if np.dot((agent_obj.position - self.position), self.velocity) >= 0 and not self.is_grounded: #if the arrow is actually coming toward us
            return True
        return False
    
    def distanceToAgent(self, agent_obj):
        return np.linalg.norm(agent_obj.position - self.position)