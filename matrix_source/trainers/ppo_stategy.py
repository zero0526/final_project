import torch
from matrix_source.agents.ppo import PPOAgent
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.utils.math_utils import to_binary
from tqdm import tqdm
import os

def compute_gae(rewards, next_values, values, dones, agent_ids, gamma, lmbda):
    """
    Generalized Advantage Estimation (GAE)
    Separated for easier inspection and algorithm flexibility.
    """
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(deltas)
    
    advantage = 0
    # Identify agent boundaries to prevent advantage bleed in multi-agent batch
    for t in reversed(range(len(deltas))):
        if t < len(deltas) - 1 and agent_ids[t] != agent_ids[t+1]:
            advantage = 0
        
        advantage = deltas[t] + gamma * lmbda * (1 - dones[t]) * advantage
        advantages[t] = advantage
    return advantages

class PPOStrategy(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.phase = 'LOWER_ONLY' # 'LOWER_ONLY', 'UPPER_ONLY', 'ALTERNATING'
        self.lower_train_num = 0
        self.upper_train_num = 0
        self.alt_train_num = 0
        self.alt_next = 'UPPER'
        self.upper_mf_ema = None
        self.mf_ema_alpha = 0.7
        
        # Hyperparams from user
        self.lower_cfg = {'min_size': 4096, 'batch': 128, 'epochs': 6}
        self.upper_cfg = {'min_size': 512, 'batch': 64, 'epochs': 4}
        self.lower_warmup_steps = 40
        self.upper_warmup_steps = 60
        self.alt_steps = 80

    def initialize_agents(self, trainer):
        # 1. Upper Agent
        trainer.shared_upper_agent = PPOAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.upper_cfg['min_size'],
            hidden_sizes=trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(trainer.config.hyper_neural['UPPER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(trainer.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=trainer.config.hyper_neural['MEMORY_SIZE'],
            batch_size=self.upper_cfg['batch'],
            k_epochs=self.upper_cfg['epochs'],
            num_instances=trainer.num_edge_agents,
            device=trainer.device
        )

        # 2. Lower Agent
        trainer.shared_lower_agent = PPOAgent(
            node_id=-1, node_type="Terminal_Group",
            state_dim=trainer.lower_state_dim,
            action_dim=trainer.lower_action_dim,
            u_action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.lower_cfg['min_size'],
            hidden_sizes=tuple(trainer.config.hyper_neural['AGENT_HIDDEN_LAYER']),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(trainer.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=trainer.config.hyper_neural['MEMORY_SIZE'],
            batch_size=self.lower_cfg['batch'],
            k_epochs=self.lower_cfg['epochs'],
            num_instances=trainer.num_terminals,
            device=trainer.device
        )

    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        
        # 1. Random actions in Lower-Only phase
        if self.phase == 'LOWER_ONLY':
            act_matrix[trainer.edge_node_ids] = torch.randint(0, 2, (len(trainer.edge_node_ids), trainer.num_services), device=trainer.device).float()
            for nid in trainer.env.static_matrices.get("cloud_ids", []):
                act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
            return act_matrix
            
        # 2. Get Observed Mean Field and Update EMA (Upper only)
        mf_global = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        if self.upper_mf_ema is None:
            self.upper_mf_ema = mf_global.clone()
        else:
            self.upper_mf_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * mf_global
            
        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = self.upper_mf_ema[trainer.edge_node_ids]
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
        
        # Use stochastic (det=False) in UPPER_ONLY phase.
        # In ALTERNATING phase, both are stochastic (joint training).
        is_det = (self.phase == 'EVAL')
        
        batch_a_ids = trainer.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, agent_indices=instance_indices, deterministic=is_det
        )
        
        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)
        
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field']
        meta = trainer.env.metadata
        
        data_sizes = batch_sizes * meta['service_input_size'][s_idx].squeeze(-1)
        s_tasks = torch.stack([data_sizes, tasks_min_accuracy, task_deadlines, meta['service_omega'][s_idx].squeeze(-1)], dim=1).float()
        s_backlogs = obs_dict['backlog'][:, s_idx].T
        s_cpus = obs_dict['cpu_alloc'][:, s_idx].T
        states = torch.cat([s_tasks, s_backlogs, s_cpus], dim=1)
        
        states[:, 0] /= trainer.config.norm_data_size
        states[:, 1] /= 100.0
        if states.shape[1] > 4:
            states[:, 4:4+2*trainer.num_nodes] /= trainer.config.norm_gflop
            
        masks = torch.ones((len(t_idx), trainer.num_nodes * trainer.max_models), device=trainer.device)
        mfs = mf_terminals[t_idx]
        
        # "Freeze" lower (argmax) only during Upper-Only phase. In Alternating, both are stochastic.
        is_det = (self.phase == 'UPPER_ONLY' or self.phase == 'EVAL')
        
        batch_actions = trainer.shared_lower_agent.choose_action_batch(
            states, mfs, masks_batch=masks, 
            agent_indices=torch.arange(trainer.num_terminals, device=trainer.device),
            deterministic=is_det
        )
        
        a_ids = torch.tensor(batch_actions, device=trainer.device)
        return a_ids // trainer.max_models, a_ids % trainer.max_models

    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        from matrix_source.trainers.train import log_transform
        
        # 1. Build states for MF network training (even if frozen, we might want to track MF loss)
        def build_state(obs, tidx, sidx):
            st = torch.cat([obs['task_reqs'][tidx], obs['backlog'][:, sidx].T, obs['cpu_alloc'][:, sidx].T], dim=1)
            st[:, 0] /= trainer.config.norm_data_size
            st[:, 1] /= 100.0
            if st.shape[1] > 4: st[:, 4:4+2*trainer.num_nodes] /= trainer.config.norm_gflop
            return st

        c_obs, n_obs = current_res['obs'], next_res['obs']
        states = build_state(c_obs, t_idx, s_idx)
        next_states = build_state(n_obs, t_idx, s_idx)
        
        # 2. Extract metrics and rewards
        reward = next_res['reward']
        rew_divisor = trainer.config.norm_lower_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        
        # 3. Handle MF training and transition storage (only if NOT frozen)
        avg_mf_loss = None
        is_frozen = (self.phase == 'UPPER_ONLY')
        
        if not is_frozen:
            done = torch.tensor([next_res["new_frame"]]*len(t_idx), dtype=torch.float32, device=trainer.device)
            c_mf, n_mf = current_res['mean_field'], next_res['mean_field']
            rewards = torch.full((len(t_idx),), norm_rew, dtype=torch.float32, device=trainer.device)
            a_ids = (n_idx * trainer.max_models + m_idx).long()
            
            avg_mf_loss = trainer.shared_lower_agent.store_transition_train_mf_batch(
                states, c_mf[t_idx], n_mf[t_idx], a_ids, rewards, next_states, done, agent_ids=t_idx
            )
            
        # 4. ALWAYS record metrics!
        trainer.aggregator.add_lower(next_res, mf_loss=avg_mf_loss, state=states[0] if len(states) > 0 else None)

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        from matrix_source.trainers.train import log_transform
        
        # 1. Extract global metrics
        reward = next_res['reward_global']
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        
        # 2. Handle MF training and transition storage (if NOT frozen)
        avg_mf_loss = None
        is_frozen = (self.phase == 'LOWER_ONLY')
        
        edge_states = s_all[trainer.edge_node_ids]
        
        if not is_frozen:
            edge_next_states = ns_all[trainer.edge_node_ids]
            dones = torch.full((trainer.num_edge_agents,), 1.0 if is_done else 0.0, dtype=torch.float32, device=trainer.device)

            # Use same EMA consistency logic for storage
            next_raw_mf = next_res['mean_fields']
            if self.upper_mf_ema is None:
                self.upper_mf_ema = next_raw_mf
            next_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * next_raw_mf
            
            edge_c_mfs = self.upper_mf_ema[trainer.edge_node_ids]
            edge_n_mfs = next_ema[trainer.edge_node_ids]
            edge_acts = acts_matrix[trainer.edge_node_ids]
            
            pw2 = 2 ** torch.arange(trainer.num_services - 1, -1, -1, device=trainer.device).float()
            edge_a_ids = (edge_acts * pw2).sum(dim=1).long()
            
            rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
            instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)

            avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
                edge_states, edge_c_mfs, next_raw_mf[trainer.edge_node_ids], edge_a_ids, rewards, edge_next_states, dones, agent_ids=instance_indices
            )
            
        # 3. ALWAYS record metrics!
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=edge_states[0] if len(edge_states) > 0 else None)

    def run_training(self, trainer):
        max_slots = trainer.env.time_manager.max_steps
        ep = 0

        pbar = tqdm(total=self.lower_warmup_steps + self.upper_warmup_steps + self.alt_steps, desc="Overall Training Progress")
        
        while True:
            if self.phase == 'ALTERNATING' and self.alt_train_num >= self.alt_steps:
                print(f"\n[Curriculum] Phase 3 Complete. Training Finished.")
                break
                
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            self.upper_mf_ema = None # Reset EMA for new episode
            current_upper_state = self.build_upper_state(trainer, obs_upper) 
            
            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    n_idx, m_idx = self.get_lower_actions(trainer, prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    self.store_lower_transitions(trainer, prev_lower_res, results, t_idx, s_idx, n_idx, m_idx)
                    
                    trainer.aggregator.add_step_matrices(
                        f_alloc=trainer.env.engine.cpu_alloc_matrix,
                        arrivals=results['info']['arrival_matrix'],
                        backlog=trainer.env.engine.backlog_queue.sum(dim=-1)
                    )
                    
                    prev_lower_res = results
                    trainer.total_lower_steps += 1 
                    
                    # 1. Train Lower Level (ONLY in Phase 1)
                    if self.phase == 'LOWER_ONLY':
                        loss = trainer.shared_lower_agent.learn(torch.arange(trainer.num_terminals, device=trainer.device))
                        if loss is not None:
                            self.lower_train_num += 1
                            trainer.aggregator.record_td_losses(lower_losses=loss)
                            
                            # Checkpoint
                            if self.lower_train_num % 10 == 0:
                                os.makedirs('checkpoints', exist_ok=True)
                                trainer.shared_lower_agent.save(f'{trainer.config.checkpoints}/ppo_lower_{self.lower_train_num}.pth')

                            # Phase Transition
                            if self.phase == 'LOWER_ONLY' and self.lower_train_num >= self.lower_warmup_steps:
                                os.makedirs('checkpoints', exist_ok=True)
                                trainer.shared_lower_agent.save(f'{trainer.config.checkpoints}/ppo_lower_phase1.pth')
                                self.phase = 'UPPER_ONLY'
                                trainer.shared_upper_agent.memory.clear()
                                print(f"\n[Curriculum] Phase 1 Complete. Switching to {self.phase}")
                            pbar.update(1)
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    trainer.aggregator.add_upper(res_upper)
                    is_ep_done = (slot == max_slots - 1)
                    
                    self.store_upper_transitions(trainer, current_upper_state, next_upper_state, obs_upper, res_upper, u_acts_matrix, is_ep_done)
                    trainer.total_upper_steps += 1 

                    # 2. Train Upper Level (ONLY in Phase 2)
                    if self.phase == 'UPPER_ONLY':
                        loss = trainer.shared_upper_agent.learn(torch.arange(trainer.num_edge_agents, device=trainer.device))
                        if loss is not None:
                            print("trainning uppper")
                            self.upper_train_num += 1
                            trainer.aggregator.record_td_losses(upper_losses=loss)
                            
                            # Checkpoint
                            if self.upper_train_num % 10 == 0:
                                os.makedirs('checkpoints', exist_ok=True)
                                trainer.shared_upper_agent.save(f'{trainer.config.checkpoints}/ppo_upper_{self.upper_train_num}.pth')

                            # Phase Transition
                            if self.phase == 'UPPER_ONLY' and self.upper_train_num >= self.upper_warmup_steps:
                                os.makedirs('checkpoints', exist_ok=True)
                                trainer.shared_upper_agent.save(f'{trainer.config.checkpoints}/ppo_upper_phase2.pth')
                                self.phase = 'ALTERNATING'
                                trainer.shared_upper_agent.memory.clear()
                                trainer.shared_lower_agent.memory.clear()
                                print(f"\n[Curriculum] Phase 2 Complete. Switching to {self.phase}")
                                # Apply Phase 3 overrides: 0.5x LR, 8000/800 buffers, 8/6 epochs
                                trainer.shared_lower_agent.set_lr_factor(0.3)
                                trainer.shared_upper_agent.set_lr_factor(0.3)
                                trainer.shared_lower_agent.k_epochs = 8
                                trainer.shared_upper_agent.k_epochs = 6
                                trainer.shared_lower_agent.min_batch_size = 8000
                                trainer.shared_upper_agent.min_batch_size = 800
                            pbar.update(1)

                    # 3. Concurrent Joint Training (Phase 3)
                    if self.phase == 'ALTERNATING':
                        # Gated update: wait until both have collected enough Phase 3 samples
                        if (all(s.size >= 8000 for s in trainer.shared_lower_agent.memory.buffers) and
                            all(s.size >= 800 for s in trainer.shared_upper_agent.memory.buffers)):
                            
                            print(f"\n[Phase 3] Concurrent Update Triggered (L: {trainer.shared_lower_agent.memory.total_size}, U: {trainer.shared_upper_agent.memory.total_size})")
                            loss_l = trainer.shared_lower_agent.learn(torch.arange(trainer.num_terminals, device=trainer.device))
                            loss_u = trainer.shared_upper_agent.learn(torch.arange(trainer.num_edge_agents, device=trainer.device))
                            
                            if loss_l is not None and loss_u is not None:
                                self.alt_train_num += 1
                                trainer.aggregator.record_td_losses(lower_losses=loss_l, upper_losses=loss_u)
                                pbar.update(1)
                                if self.alt_train_num >= self.alt_steps: break

                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
            print(f"--- Curriculum Status ---")
            print(f"Phase: {self.phase} | Lower: {self.lower_train_num}/{self.lower_warmup_steps} | Upper: {self.upper_train_num}/{self.upper_warmup_steps} | Alt: {self.alt_train_num}/{self.alt_steps}")
            ep += 1
        
        # Save final models
        os.makedirs('checkpoints', exist_ok=True)
        trainer.shared_lower_agent.save(f'{trainer.config.checkpoints}/ppo_lower_phase3.pth')
        trainer.shared_upper_agent.save(f'{trainer.config.checkpoints}/ppo_upper_phase3.pth')
        pbar.close()

    def load_checkpoints(self, trainer, lower_path=None, upper_path=None):
        """Loads checkpoints for evaluation."""
        if lower_path and os.path.exists(lower_path):
            trainer.shared_lower_agent.load(lower_path)
            print(f"[PPOStrategy] Lower agent loaded from {lower_path}")
        
        if upper_path and os.path.exists(upper_path):
            trainer.shared_upper_agent.load(upper_path)
            print(f"[PPOStrategy] Upper agent loaded from {upper_path}")

    def run_evaluation(self, trainer, num_episodes=5):
        """Runs a deterministic evaluation loop."""
        print(f"\n>>> Starting Evaluation ({num_episodes} episodes) <<<")
        self.phase = 'EVAL'
        max_slots = trainer.env.time_manager.max_steps
        
        for ep in range(num_episodes):
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            self.upper_mf_ema = None 
            current_upper_state = self.build_upper_state(trainer, obs_upper) 
            
            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    n_idx, m_idx = self.get_lower_actions(trainer, prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    
                    trainer.aggregator.add_lower(results)
                    prev_lower_res = results
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    trainer.aggregator.add_upper(res_upper)
                    
                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
        
        print(f"\n>>> Evaluation Complete <<<")

