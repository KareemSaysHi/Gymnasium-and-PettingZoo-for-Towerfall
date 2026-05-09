# PettingZoo Environment for Towerfall

Completed as a final project for COS435 / ECE433 - Reinforcement Learning at Princeton University.

This project builds upon Vini Ruela's [TowerFall-AI](https://github.com/TowerfallAi/towerfall-ai), which allows a Python script to interact with a TowerFall instance.  Both PettingZoo and Gymnasium environments have been made for TowerFall, and this repository has examples for training with both environments.  Large credit goes to him for the foundational code, this project would not be possible without it.

### Running the environments
**To run this code, you must already own a copy of TowerFall.**  Go buy it, it's an awesome game :)
- Download the latest release of [Towerfall-AI](https://github.com/TowerfallAi/towerfall-ai) and patch your copy of TowerFall
- Delete the python folder inside of towerfall-ai-main
- Download this repository, unzip the folder, and put it where the python folder originally was
See the files in the repository for details on the environments and training!

### Creating your own agents

Agents should be created for the TowerFall environment that take in an observation (described as the observation space of both environmentss) and output an action.  Actions are a MultiDiscrete(9, 4), consisting of at most one element in each of the following categories:
- Direction (the 8 directional inputs)
- Button (jump, shoot, dash)
See BaselineAgent for an example implementation
