from src.envs.environment import SixGEnvironment
from src.configs.configs import cfg
import numpy as np
from collections import defaultdict
import random
if __name__=="__main__":
    env= SixGEnvironment(cfg)
    upper_state =env.reset_upper()
    lower_state = env.reset_lower()
    action_map= {
        "N10": np.array([0,1,1,1,1]),
        "N11": np.array([1, 0, 1, 1, 1]),
        "N12": np.array([1, 1, 0, 1, 1]),
        "N13": np.array([1, 1, 1, 0, 1]),
        "N15": np.array([1, 1, 1, 1, 0]),
        "N4": np.array([0, 1, 1, 1, 1]),
        "N5": np.array([1, 0, 1, 1, 1]),
        "N8": np.array([1, 1, 0, 1, 1]),
    }
    services= {s.get("id"): len(s.get("models"))-1 for s in cfg.services.values()}
    max_service= max(services.values())
    service_node= defaultdict(list)
    for k,v in action_map.items():
        for i in range(v.shape[0]):
            if i not in service_node[k]:
                service_node[k] = []
            service_node[i].append(k)
    upper_state= env.step_upper(action_map)
    lower_state= env.step_lower([])
    lower_action_map= []
    for task,backlog, cpu, mean_field  in lower_state.get('next_states').values():
        lower_action_map.append((task, random.choice(service_node[task.service_id]), random.randint(0, services[task.service_id]) ))
    print(f"num of task {len(lower_action_map)}")
    env.step_lower(lower_action_map)
    print("hello")

