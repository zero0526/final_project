import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from matrix_source.envs.matrix_env import MatrixSixGEnvironment
from matrix_source.envs.workload_generator import MatrixWorkloadGenerator
from matrix_source.agents.d3qn import D3QNAgent
from matrix_source.configs.configs import cfg
from matrix_source.utils.math_utils import to_binary, one_hot
from matrix_source.visualize.aggregator import MetricsAggregator

class Trainer:
    def __init__(self):
        self.config = cfg
        self.device = cfg.hyper_neural.get('DEVICE', 'cpu')
        
        # 1. Initialize Environment & Workload
        self.env = MatrixSixGEnvironment(config=cfg, device=self.device)
        self.workload_gen = MatrixWorkloadGenerator(cfg, self.env.metadata, device=self.device)

        self.num_services = self.env.engine.num_services
        self.num_nodes = self.env.engine.num_nodes
        self.num_terminals = self.workload_gen.num_terminals
        self.max_models = self.env.metadata.get("max_models", 5)

        # State Dims
        self.upper_state_dim = (self.num_services * 2)
        self.lower_state_dim = 4 + (self.num_nodes * 2) + self.num_nodes # Added terminal-to-node one-hot map

        self.upper_action_dim = self.num_services
        self.upper_u_action_dim = 1 << self.num_services
        self.lower_action_dim = self.num_nodes + self.max_models
        self.lower_u_action_dim = self.num_nodes * self.max_models

        # --- Hyperparams ---
        self.min_epsilon = cfg.hyper_neural.get("EPSILON", 0.05)
        self.epsilon_decay = cfg.hyper_neural.get("EPSILON_DECAY", 0.9985)
        self.epsilons = {nid: 1.0 for nid in range(self.num_nodes)}
        self.lower_epsilons = {tid: 1.0 for tid in range(self.num_terminals)}
        self.zeta_initial = cfg.hyper_neural.get("ZETA", 1.0)
        self.zeta_max = cfg.hyper_neural.get("ZETA_MAX", 10.0)
        self.zeta_upper = self.zeta_initial
        self.zeta_lower = self.zeta_initial

        # Training control variables
        self.total_lower_steps = 0
        self.total_upper_steps = 0
        self.lower_stable_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][0]*10
        self.lower_start_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][1]
        
        self.aggregator = MetricsAggregator()
        self.shared_upper_agent: D3QNAgent = None
        self.shared_lower_agent: D3QNAgent = None
        
        # Track which nodes are edge agents (to map to weight indices)
        self.edge_node_ids = [nid for nid in range(self.num_nodes) 
                             if nid not in self.env.static_matrices.get("cloud_ids", [])]
        self.node_to_instance = {nid: i for i, nid in enumerate(self.edge_node_ids)}
        self.num_edge_agents = len(self.edge_node_ids)

        self.__init_agents()

    def __init_agents(self):
        # 1. Initialize Shared Upper Agent (for all edge nodes)
        self.shared_upper_agent = D3QNAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=self.upper_state_dim,
            action_dim=self.upper_action_dim,
            u_action_dim=self.upper_u_action_dim,
            mf_hidden_sizes=tuple(self.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(self.config.hyper_neural['MF_LR']),
            buffer_min_size= float(self.config.hyper_neural["BUFFER_MIN_SIZE"][0]),
            hidden_sizes=self.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(self.config.hyper_neural['UPPER_LR']),
            gamma=self.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(self.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=self.config.hyper_neural['MEMORY_SIZE'],
            batch_size=self.config.hyper_neural['BATCH_SIZE'],
            num_instances=self.num_edge_agents, # Unified independent parameters for all edge nodes
            device=self.device
        )

        self.shared_lower_agent = D3QNAgent(
            node_id=-1, node_type="Terminal_Group",
            state_dim=self.lower_state_dim,
            action_dim=self.lower_action_dim,
            u_action_dim=self.lower_u_action_dim,
            mf_hidden_sizes=tuple(self.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(self.config.hyper_neural['MF_LR']),
            buffer_min_size=float(self.config.hyper_neural["BUFFER_MIN_SIZE"][1]),
            hidden_sizes=tuple(self.config.hyper_neural['AGENT_HIDDEN_LAYER']),
            lr=float(self.config.hyper_neural['LOWER_LR']),
            gamma=self.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(self.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=self.config.hyper_neural['MEMORY_SIZE'],
            batch_size=self.config.hyper_neural['BATCH_SIZE'],
            num_instances=self.num_terminals, # Independent weights for every terminal
            device=self.device
        )

    def train(self):
        num_eps = self.config.hyper_neural['NUMOF_TRAIN_EP']
        max_slots = self.env.time_manager.max_steps

        for ep in tqdm(range(num_eps), desc="Training"):
            obs = self.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            
            current_upper_state = self.get_upper_state(obs_upper) 
            
            for slot in range(max_slots):
                if self.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(current_upper_state, obs_upper)
                    self.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = self.workload_gen.generate_step()
                if len(t_idx) > 0:
                    n_idx, m_idx = self.get_lower_actions(prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    results = self.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    self.store_lower_transitions(prev_lower_res, results, t_idx, s_idx, n_idx, m_idx)
                    
                    # Accumulate Metrics
                    self.aggregator.add_step_matrices(
                        f_alloc=self.env.engine.cpu_alloc_matrix,
                        arrivals=results['info']['arrival_matrix'],
                        backlog=self.env.engine.backlog_queue.sum(dim=-1)
                    )
                    
                    prev_lower_res = results
                    
                    # train lower
                    loss = self.shared_lower_agent.learn(torch.arange(self.num_terminals, device=self.device))
                    if loss is not None:
                        self.total_lower_steps += 1 # Count samples collected
                        self.aggregator.record_td_losses(lower_losses=loss)
                else:
                    self.env.time_manager.tick()

                if self.env.time_manager.is_new_frame():
                    res_upper = self.env.collect_upper_metrics()
                    next_upper_state = self.get_upper_state(res_upper)
                    
                    self.aggregator.add_upper(res_upper)

                    is_ep_done = (slot == max_slots - 1)
                    
                    # Phased Curriculum: Only collect and train Upper after Lower is stable
                    if self.total_lower_steps >= self.lower_stable_threshold/10:
                        self.store_upper_transitions(current_upper_state, next_upper_state, obs_upper, res_upper, u_acts_matrix, is_ep_done)

                    # train upper
                    loss = self.shared_upper_agent.learn(torch.arange(self.num_edge_agents, device=self.device))
                    if loss is not None:
                        self.total_upper_steps += 1 # Approximate upper samples
                        self.aggregator.record_td_losses(upper_losses=loss)

                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            # update ep and history
            self.update_rates(ep)
            self.aggregator.store_history()
            self.aggregator.report_episode(ep)
            print(f"--- Global Metrics ---")
            print(f"Lower Samples: {self.total_lower_steps} | Upper Samples: {self.total_upper_steps}")
            print(f"Zeta Lower: {self.zeta_lower:.4f} | Zeta Upper: {self.zeta_upper:.4f}")
            print(f"Current Epsilon (Edge N0): {self.epsilons[0]:.4f}")

    def get_upper_state(self, obs_upper):
        actions = obs_upper['actions'] # (N, S)
        phi = obs_upper['phi_prob']    # (N, S)
        
        state = torch.cat([actions, phi], dim=-1)
        return state

    def get_upper_actions(self, current_upper_state, obs_upper):
        act_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        
        # Batch preparation for edge nodes
        edge_states = current_upper_state[self.edge_node_ids]
        edge_mfs = mf_global[self.edge_node_ids]
        
        # Map edge node IDs to instance indices (0 to num_edge_agents-1)
        instance_indices = torch.tensor([self.node_to_instance[nid] for nid in self.edge_node_ids], device=self.device)
        
        # Vectorized batch inference for all edge agents
        batch_a_ids = self.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, self.zeta_upper, agent_indices=instance_indices
        )
        
        for i, nid in enumerate(self.edge_node_ids):
            a_id = batch_a_ids[i]
            act_matrix[nid] = torch.tensor(to_binary(a_id, self.num_services), device=self.device)
        
        # Cloud nodes always use all services
        for nid in self.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(self.num_services, device=self.device)
        return act_matrix

    def get_lower_actions(self, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field']
        
        meta = self.env.metadata
        unit_sizes = meta['service_input_size']
        service_omega = meta['service_omega']
        placement_matrix = self.env.engine.placement_matrix
        model_accs = meta['model_accuracies']

        num_reqs = len(t_idx)
        if num_reqs == 0:
            return torch.tensor([], dtype=torch.long, device=self.device), torch.tensor([], dtype=torch.long, device=self.device)

        # 1. Vectorized State Construction
        data_sizes = batch_sizes * unit_sizes[s_idx].squeeze(-1)
        s_tasks = torch.stack([
            data_sizes, 
            tasks_min_accuracy, 
            task_deadlines, 
            service_omega[s_idx].squeeze(-1)
        ], dim=1).float() # (Batch, 4)
        
        s_backlogs = obs_dict['backlog'][:, s_idx].T # (Batch, num_nodes)
        s_cpus = obs_dict['cpu_alloc'][:, s_idx].T # (Batch, num_nodes)
        
        # Add Terminal-to-Node mapping (One-Hot)
        s_terminal_map = self.env.engine.terminal_to_node_map[t_idx] # (Batch, num_nodes)
        
        states = torch.cat([s_tasks, s_backlogs, s_cpus, s_terminal_map], dim=1) # (Batch, lower_state_dim)
        
        # datasize, min_accuracy, deadline, service_omega
        states[:, 0] /= self.config.norm_data_size
        states[:, 1] /= 100.0
        # states[:, 2] /= float(norm_factors.get("2", 1.0))
        # states[:, 3] /= float(norm_factors.get("3", 1.0))
        
        # Broadcast normalization for all unknown topology dimensions
        if states.shape[1] > 4:
            states[:, 4:4+2*self.num_nodes] /= self.config.norm_gflop # Standard Backlog Divisor
        # p_mask = torch.zeros_like(placement_matrix[:, s_idx])
        # # placement_matrix[:, s_idx].T > 0 # (Batch, num_nodes)
        # a_mask = model_accs[s_idx, :] >= tasks_min_accuracy.unsqueeze(1)  # (Batch, max_models)
        # masks = (p_mask.unsqueeze(2) & a_mask.unsqueeze(1)).flatten(start_dim=1).float()  # (Batch, action_dim)
        #
        # # Fallback if entirely masked
        # invalid_mask_rows = (masks.sum(dim=1) == 0)
        # masks[invalid_mask_rows] = 1.0
        # 2. Vectorized Masks - DISABLED (Agent must learn topology independently)
        masks = torch.ones((len(t_idx), self.num_nodes * self.max_models), device=self.device)
        
        # 3. Vectorized mean fields
        mfs = mf_terminals[t_idx]
        
        # 4. Batch Inference
        batch_actions = self.shared_lower_agent.choose_action_batch(
            states, mfs, self.zeta_lower, masks_batch=masks.to(self.device), agent_indices=torch.arange(self.num_terminals, device=self.device)
        )
        
        a_ids = torch.tensor(batch_actions, device=self.device)
        node_indices = a_ids // self.max_models
        model_indices = a_ids % self.max_models
        
        return node_indices, model_indices

    def store_lower_transitions(self, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        if len(t_idx) == 0: return
        
        reward = next_res['reward']
        done = torch.tensor([next_res["new_frame"]]*len(t_idx), dtype=torch.float32, device=self.device)
        
        c_obs, n_obs = current_res['obs'], next_res['obs']
        c_mf, n_mf = current_res['mean_field'], next_res['mean_field']
        
        s_tasks_current = c_obs['task_reqs'][t_idx] # (Batch, 4)
        s_backlogs_current = c_obs['backlog'][:, s_idx].T
        s_cpus_current = c_obs['cpu_alloc'][:, s_idx].T
        s_term_map = self.env.engine.terminal_to_node_map[t_idx]
        states = torch.cat([s_tasks_current, s_backlogs_current, s_cpus_current, s_term_map], dim=1)
        
        s_tasks_next = n_obs['task_reqs'][t_idx]
        s_backlogs_next = n_obs['backlog'][:, s_idx].T
        s_cpus_next = n_obs['cpu_alloc'][:, s_idx].T
        next_states = torch.cat([s_tasks_next, s_backlogs_next, s_cpus_next, s_term_map], dim=1)
        

        states[:, 0] /= self.config.norm_data_size
        states[:, 1] /= 100.0
        # states[:, 2] /= float(norm_factors.get("2", 1.0))
        # states[:, 3] /= float(norm_factors.get("3", 1.0))
        
        next_states[:, 0] /= self.config.norm_data_size
        next_states[:, 1] /= 100.0
        # next_states[:, 2] /= float(norm_factors.get("2", 1.0))
        # next_states[:, 3] /= float(norm_factors.get("3", 1.0))
        
        if states.shape[1] > 4:
            states[:, 4:4+2*self.num_nodes] /= self.config.norm_gflop
            next_states[:, 4:4+2*self.num_nodes] /= self.config.norm_gflop
     
        # Normalize reward
        rew_divisor = self.config.norm_lower_rw
        normalized_reward = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        rewards = torch.full((len(t_idx),), normalized_reward, dtype=torch.float32, device=self.device)
        
        a_ids = (n_idx * self.max_models + m_idx).long()
        
        c_mfs_batch = c_mf[t_idx]
        n_mfs_batch = n_mf[t_idx]
        
        avg_mf_loss = self.shared_lower_agent.store_transition_train_mf_batch(
            states, c_mfs_batch, n_mfs_batch, a_ids, rewards, next_states, done, agent_ids=t_idx
        )
        
        sample_state = states[0] if len(states) > 0 else None
        self.aggregator.add_lower(next_res, mf_loss=avg_mf_loss, state=sample_state)

    def store_upper_transitions(self, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        reward = next_res['reward_global']
        c_mf = current_res.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        n_mf = next_res['mean_fields']
        
        done_val = 1.0 if is_done else 0.0
        dones = torch.full((self.num_edge_agents,), done_val, dtype=torch.float32, device=self.device)

        # Batch preparation for edge agents
        edge_states = s_all[self.edge_node_ids]
        edge_next_states = ns_all[self.edge_node_ids]
        edge_c_mfs = edge_c_mfs = c_mf[self.edge_node_ids]
        edge_n_mfs = n_mf[self.edge_node_ids]
        edge_acts = acts_matrix[self.edge_node_ids]
        
        # Vectorized conversion of binary actions to action IDs
        powers_of_2 = 2 ** torch.arange(self.num_services - 1, -1, -1, device=self.device).float()
        edge_a_ids = (edge_acts * powers_of_2).sum(dim=1).long()

        # Normalize global reward
        rew_divisor = self.config.norm_upper_rw
        normalized_reward = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        rewards = torch.full((self.num_edge_agents,), normalized_reward, dtype=torch.float32, device=self.device)

        # Map edge node IDs to instance indices
        instance_indices = torch.tensor([self.node_to_instance[nid] for nid in self.edge_node_ids], device=self.device)

        # Unified batch transition storage and MF training
        avg_mf_loss = self.shared_upper_agent.store_transition_train_mf_batch(
            edge_states, edge_c_mfs, edge_n_mfs, edge_a_ids, rewards, edge_next_states, dones, agent_ids=instance_indices
        )
        
        # Log metrics using first edge node state as sample
        sample_state = edge_states[0] if len(edge_states) > 0 else None
        self.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=sample_state)

    def update_rates(self, ep):
        # 1. Update Epsilons
        for nid in self.epsilons: 
            self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay)
        for tid in self.lower_epsilons: 
            self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay)
            
        # 2. Simple Linear Zeta Annealing
        self.zeta_lower = min(self.zeta_max, self.zeta_initial + self.total_lower_steps * self.config.zeta_lower_step)
        self.zeta_upper = min(self.zeta_max, self.zeta_initial + self.total_upper_steps * self.config.zeta_upper_step)

def log_transform(reward: float) -> float:
    return reward

if __name__ == "__main__":
    Trainer().train()