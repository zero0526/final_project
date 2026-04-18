from src.envs import SixGEnvironment, Task
from src.agents import D3QNAgent
from src.configs import cfg
from src.utils import one_hot, to_binary, from_binary
from src.visualize.aggregator import MetricsAggregator

from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
import math
from tqdm import tqdm
import sys


class Trainer:
    def __init__(self):
        self.config = cfg
        self.env = SixGEnvironment(num_terminals=cfg.hyper_neural["NUM_LOWER_AGENTS"], config=cfg)

        self.upper_agents: Dict[str, D3QNAgent] = {}
        self.lower_agents: Dict[str, D3QNAgent] = {}

        self.upper_state_dim = self.env.num_services * 2
        # State: omega, data_size, deadline, accuracy (4) + Service Backlogs (N) + Service CPU (N) + Global Backlogs (N) + Global CPU (N)
        self.lower_state_dim = 4 + len(self.env.computing_nodes) * 4
        self.upper_u_action_dim = 1 << self.env.num_services
        self.upper_action_dim = self.env.num_services
        self.lower_action_dim = len(self.env.computing_nodes) + self.env.max_models_total
        self.lower_u_action_dim = len(self.env.computing_nodes) * self.env.max_models_total

        # --- INITIALIZE EPSILON & ZETA ---
        self.min_epsilon = cfg.hyper_neural.get("EPSILON", 0.05)
        self.epsilon_decay_factor = cfg.hyper_neural.get("EPSILON_DECAY", 0.9985)

        self.epsilons = {nid: 1.0 for nid in self.env.agent_node_ids}
        self.lower_epsilons = {tid: 1.0 for tid in self.env.terminals.keys()}

        self.zeta = cfg.hyper_neural.get("ZETA", 1.0)

        self.service_id_node: Dict[int, List[str]] = {}
        self.aggregator = MetricsAggregator()

        self.__init_agents()

    def __init_agents(self):
        for i in self.env.agent_node_ids:
            self.upper_agents[i] = D3QNAgent(
                state_dim=self.upper_state_dim,
                action_dim=self.upper_action_dim,
                u_action_dim= self.upper_u_action_dim,
                mf_hidden_sizes=tuple(self.config.hyper_neural["MF_HIDDEN_LAYER"]),
                mf_lr=float(self.config.hyper_neural['MF_LR']),
                hidden_sizes=self.config.hyper_neural['AGENT_HIDDEN_LAYER'],
                lr=float(self.config.hyper_neural['UPPER_LR']),
                gamma=self.config.hyper_neural['DISCOUNT_FACTOR'],
                alpha=float(self.config.hyper_neural['UPDATE_TARGET_COEF']),
                buffer_size=self.config.hyper_neural['MEMORY_SIZE'],
                batch_size=self.config.hyper_neural['BATCH_SIZE'],
                exclude_zero=True
            )

        for t in self.env.terminals.keys():
            self.lower_agents[t] = D3QNAgent(
                state_dim=self.lower_state_dim,
                action_dim=self.lower_action_dim,
                u_action_dim=self.lower_u_action_dim,
                mf_hidden_sizes=tuple(self.config.hyper_neural["MF_HIDDEN_LAYER"]),
                mf_lr=float(self.config.hyper_neural['MF_LR']),
                hidden_sizes=tuple(self.config.hyper_neural['AGENT_HIDDEN_LAYER']),
                lr=float(self.config.hyper_neural['LOWER_LR']),
                gamma=self.config.hyper_neural['DISCOUNT_FACTOR'],
                alpha=float(self.config.hyper_neural['UPDATE_TARGET_COEF']),
                buffer_size=self.config.hyper_neural['MEMORY_SIZE'],
                batch_size=self.config.hyper_neural['BATCH_SIZE']
            )

    def update_exploration_rates(self, ep):
        # Apply epsilon decay as usual
        for nid in self.epsilons:
            self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay_factor)
        for tid in self.lower_epsilons:
            self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay_factor)
            
        # Linear growth for Zeta (Inverse Temperature) based on ANNEALING_LENGTH
        annealing_len = self.config.hyper_neural.get("ANNEALING_LENGTH", 1500)
        start_zeta = self.config.hyper_neural.get("ZETA", 1.0)
        target_zeta = 15.0 # High value for strong exploitation
        
        if ep < annealing_len:
            self.zeta = start_zeta + (target_zeta - start_zeta) * (ep / annealing_len)
        else:
            self.zeta = target_zeta

    def train(self):
        num_eps = self.config.hyper_neural['NUMOF_TRAIN_EP']
        upper_state = self.env.reset_upper()
        lower_state = self.env.reset_lower()

        pbar = tqdm(range(num_eps), desc="Training")
        for ep in pbar:
            self.update_exploration_rates(ep)

            while True:
                if lower_state.get("new_frame"):
                    upper_actions, prev_states, prev_mfs = self.placement_service(upper_state)
                    self.service_map_node(upper_actions)
                    upper_state = self.env.step_upper(upper_actions)
                    self.aggregator.add_upper(upper_state)
                    
                    curr_mfs = self.add_state2buffer(prev_states, prev_mfs, upper_actions, upper_state)

                    for nid, u_agent in self.upper_agents.items():
                        u_agent.learn_mf(prev_states[nid], prev_mfs[nid], curr_mfs[nid])
                        u_agent.learn()

                assigned_tasks, prev_states, prev_mfs = self.task_scheduling(lower_state)
                lower_state = self.env.step_lower(assigned_tasks)
                self.aggregator.add_lower(lower_state)
                
                curr_lower_mfs = self.add_lower_action(assigned_tasks, prev_states, prev_mfs, lower_state)

                for tid, t_agent in self.lower_agents.items():
                    t_agent.learn_mf(prev_states[tid], prev_mfs[tid], curr_lower_mfs[tid])
                    t_agent.learn()
                
                if lower_state.get("done"):
                    break
            
            # Federated Edge Aggregation (SCAFFOLD + HierFAVG)
            self.edge_aggregation()
            
            self.aggregator.store_history()
            self.aggregator.report_episode(
                ep, 
                success_counts=self.env.episode_success_counts, 
                failure_counts=self.env.episode_failure_counts
            )
            
            # Update plots and display every 300 episodes
            if (ep + 1) % 300 == 0:
                self.aggregator.plot_history(ep=ep+1)
                try:
                    from IPython.display import display, Image
                    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
                        display(Image(filename="src/visualize/plots/training_metrics.png"))
                except ImportError:
                    pass
            else:
                # Still store history every episode for moving average consistency
                pass
            
            pbar.set_postfix({
                "reward": f"{self.aggregator.history['total_reward'][-1]:.2f}",
                "zeta": f"{self.zeta:.2f}"
            })
            

            upper_state= self.env.reset_upper()
            lower_state = self.env.reset_lower()

    def edge_aggregation(self):
        lr = self.config.hyper_neural['LOWER_LR']
        for edge_id, terminals in self.env.terminals_group.items():
            if not terminals: continue
            
            t_ids = [t.id for t in terminals]
            agents = [self.lower_agents[tid] for tid in t_ids]
            n_agents = len(agents)
            
            # Aggregate y_i (current local base params) -> x^+
            global_base_params = []
            for i in range(len(agents[0].eval_net.get_base_params())):
                avg_p = sum(a.eval_net.get_base_params()[i].data.clone() for a in agents) / n_agents
                global_base_params.append(avg_p)
                
            # Aggregate c_i -> c_edge^+
            c_edge_new = []
            for i in range(len(agents[0].c_i)):
                c_edge_new.append(sum(a.c_i[i].clone() for a in agents) / n_agents)
                
            # Update c_i for each agent and load new global params
            for agent in agents:
                local_base = agent.eval_net.get_base_params()
                init_base = agent.initial_base_params
                K = agent.steps_in_round if agent.steps_in_round > 0 else 1
                
                for i in range(len(local_base)):
                    # Exact SCAFFOLD update: c_i^+ = c_i - c_edge + \frac{1}{K} \sum raw_gradient
                    avg_raw_grad = agent.grad_sum[i] / K
                    delta_c = avg_raw_grad
                    agent.c_i[i] = agent.c_i[i] - agent.c_edge[i] + delta_c
                    
                    # Load glob_p
                    local_base[i].data.copy_(global_base_params[i])
                    # Update target net as well since base is shared
                    target_base = agent.target_net.get_base_params()
                    target_base[i].data.copy_(global_base_params[i])
                
                # Setup c_edge and initial_base for next round
                agent.c_edge = [ce.clone() for ce in c_edge_new]
                agent.save_base_initial()

    def placement_service(self, upper_state):
        all_states = upper_state.get("next_states")
        all_mfs = upper_state.get("mean_fields")
        prev_states: Dict[str, np.ndarray] = {}
        actions: Dict[str, np.ndarray] = {}

        for node in self.env.computing_nodes:
            nid = node.id
            if nid ==self.env.cloud_node_id: continue
            node_placement, node_phi = all_states[nid]
            state = np.concatenate([node_placement, node_phi], axis=-1)
            prev_states[nid] = state
            mf = all_mfs[nid]

            action_id = self.upper_agents[nid].choose_action(state, mf, self.epsilons[nid], self.zeta)
            action_vec = to_binary(action_id, self.upper_action_dim)
            actions[nid] = action_vec

        return actions, prev_states, all_mfs,

    def add_state2buffer(self, prev_states, prev_mfs, actions, upper_state):
        curr_states = upper_state.get("next_states")
        curr_mfs = upper_state.get("mean_fields")
        local_penalties = upper_state.get("local_penalties", {})
        global_reward = upper_state.get("reward", 0)
        
        # Scale for local penalty: approx 0.01 per dropped task
        local_penalty_scale = self.config.hyper_neural.get("OMEGA_Q1", 1.0) * 0.01

        for node in self.env.computing_nodes:
            nid = node.id
            if nid ==self.env.cloud_node_id: continue
            node_placement, node_phi = curr_states[nid]
            curr_s = np.concatenate([node_placement, node_phi], axis=-1)
            
            agent_reward = global_reward - (local_penalties.get(nid, 0) * local_penalty_scale)
            
            self.upper_agents[nid].store_transition(
                prev_states[nid], prev_mfs[nid], curr_mfs[nid],
                from_binary(actions[nid]),
                agent_reward, curr_s, upper_state.get("done")
            )
        return curr_mfs

    def service_map_node(self, actions: Dict[str, np.ndarray]):
        self.service_id_node = defaultdict(list)
        cloud_id= self.env.node_id_dict[self.env.cloud_node_id]

        for nid, action in actions.items():
            active_services = np.where(action == 1)[0]
            for i in active_services:
                self.service_id_node[i].append(self.env.node_id_dict[nid])
        for i in range(self.env.num_services):
            self.service_id_node[i].append(cloud_id)


    def task_scheduling(self, lower_state):
        states = lower_state.get("next_states")
        actions: List[Tuple[Task, int, int]] = []
        prev_states: Dict[str, np.ndarray] = {}
        prev_mfs: Dict[str, np.ndarray] = {}

        for tid, t in self.env.terminals.items():
            # states[tid] now contains 6 elements: task, svc_backlogs, svc_cpu, mf, global_backlogs, global_cpu
            task, backlogs, cpu_allocations, mf, global_backlogs, global_cpu = states[tid]

            s = self.get_lower_state(task, backlogs, cpu_allocations, global_backlogs, global_cpu)

            prev_states[tid] = s
            prev_mfs[tid] = mf
            mask = self.get_action_mask(task)
            action_id = self.lower_agents[tid].choose_action(s, mf, self.lower_epsilons[tid], self.zeta, mask, task.assigned_node_id)

            node_id, model_id = self.decode_lower_action_idx(action_id)
            actions.append((task, node_id, model_id))
            
            # Record offloading flow for reporting
            self.aggregator.episode_offloading_matrix[task.source_node_id][node_id] += 1
            
        return actions, prev_states, prev_mfs

    def get_action_mask(self, task):
        num_nodes = len(self.env.computing_nodes)
        max_models = self.env.max_models_total

        node_mask = np.zeros(num_nodes)
        placed_nodes = self.service_id_node.get(task.service_id, [])
        node_mask[placed_nodes] = 1

        action_node_mask = np.repeat(node_mask, max_models)

        model_mask_local = np.zeros(max_models)
        models = self.env.service_config[task.service_id].get("models", [])

        valid_indices = [m.get("id") for m in models if m.get("accuracy") >= task.min_accuracy]

        if not valid_indices and models:
            valid_indices = [models[-1].get("id")]

        for m_id in valid_indices:
            if m_id < max_models:
                model_mask_local[m_id] = 1

        action_model_mask = np.tile(model_mask_local, num_nodes)
        final_mask = action_node_mask * action_model_mask

        return final_mask

    def get_lower_state(self, task, backlogs, cpu_allocations, global_backlogs=None, global_cpu=None):
        norm_d = task.total_data_size_mb / 400.0
        norm_deadline = (task.deadline - task.created_at) / 7.0
        norm_acc = task.min_accuracy / 100.0
        
        # Service-specific loads
        norm_backlogs = backlogs / 1000.0
        norm_resource_allocations = cpu_allocations / 4000.0
        
        state_parts = [np.array([task.omega, norm_d, norm_deadline, norm_acc]), norm_backlogs, norm_resource_allocations]
        
        # Global node loads (cross-service awareness)
        if global_backlogs is not None and global_cpu is not None:
            norm_global_backlogs = global_backlogs / 5000.0 # Scale reflects total capacity
            norm_global_cpu = global_cpu / 15000.0
            state_parts.extend([norm_global_backlogs, norm_global_cpu])
            
        return np.concatenate(state_parts, axis=-1)

    def decode_lower_action_idx(self, lower_action_id: int):
        node_id = int(lower_action_id // self.env.max_models_total)
        model_id = int(lower_action_id % self.env.max_models_total)
        return self.env.invert_node_id[node_id], model_id

    def encode_lower_action(self, action: Tuple[int, int]):
        return self.env.max_models_total * self.env.node_id_dict[action[0]] + action[1]

    def add_lower_action(self, actions: List[Tuple[Task, int, int]], prev_states: Dict[str, np.ndarray],
                          pre_mfs: Dict[str, np.ndarray], lower_state: Dict[str, Any]):
        curr_states = lower_state.get("next_states")
        rewards: set= lower_state.get("rewards")
        curr_mfs: Dict[str, np.ndarray] = {}
        action_dict: Dict[str, Tuple[str, int]] = {action[0].terminal_id: (action[1], action[2]) for action in actions}

        # Get both service-specific virtual delays and global node backlogs
        v_delays = lower_state.get("info", {}).get("virtual_delay", {})
        global_backlogs = lower_state.get("next_states", {}).get(next(iter(self.env.terminals.keys())))[4] # Index 4 is total_backlog
        
        for tid, t in self.env.terminals.items():
            task, backlogs, cpu_allocations, curr_mf, g_backlogs, g_cpu = curr_states[tid]
            curr_mfs[tid] = curr_mf
            
            s = self.get_lower_state(task, backlogs, cpu_allocations, g_backlogs, g_cpu)
            
            target_node_id, _ = action_dict[tid]
            target_node_idx = self.env.node_id_dict[target_node_id]
            
            # Local service delay
            node_v_delay = v_delays.get(target_node_id, np.zeros(self.env.num_services))[task.service_id]
            
            # Global node load (normalized) to force awareness of other services' contention
            node_total_load = g_backlogs[target_node_idx] / 5000.0
            
            # Composite reward: service delay + global congestion penalty
            rw = - (float(node_v_delay) + 0.5 * node_total_load)
            
            # Extra penalty if the task was dropped or failed QoS immediately
            if tid in rewards:
                penalty = self.config.hyper_neural.get("OMEGA_Q1", 1.0) * 20.0 
                rw -= penalty
                
            self.lower_agents[tid].store_transition(
                prev_states[tid],
                pre_mfs[tid],
                curr_mf,
                self.encode_lower_action(action_dict[tid]),
                rw,
                s,
                lower_state.get("done")
            )
        return curr_mfs

if __name__ == "__main__":
    trainer = Trainer()
    trainer.env.step_upper({trainer.env.invert_node_id[2]: np.array([1,1,1,0])})
    trainer.train()