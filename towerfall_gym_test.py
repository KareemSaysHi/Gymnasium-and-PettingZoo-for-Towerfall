from gymnasium.utils.env_checker import check_env
from towerfall_env_gym import TowerFallEnvGym

# This will catch many common issues
try:
    check_env(TowerFallEnvGym())
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")