from src.envs.environment import SixGEnvironment
from src.configs.configs import cfg
import numpy as np

if __name__=="__main__":
    env= SixGEnvironment(cfg)
    upper_state =env.reset_upper()
    lower_state = env.reset_lower()
    action_map= {
        "N10": np.array([1,1,1,1]),
        "N11": np.array([1, 1, 1, 1]),
        "N12": np.array([1, 1, 1, 1]),
        "N13": np.array([1, 1, 1, 1]),
        "N15": np.array([1, 1, 1, 1]),
        "N4": np.array([1, 1, 1, 1]),
        "N5": np.array([1, 1, 1, 1]),
        "N8": np.array([1, 1, 1, 1]),
    }
    upper_state= env.step_upper(action_map)
    lower_state= env.step_lower(action_map)