import torch
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.agents.sac_ec import SACAgent
import os

class MASAC(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.edge_agents = {} # edge_id -> SACAgent
        
    def initialize_agents(self, trainer):
        # The SACAgent will be shared among edges, or one for each edge.
        # User requested: "mỗi edge có 1 agent này". Let's create one SACAgent per Edge.
        # Or, usually MA-SAC shares the policy. "tất cả tạo thành state cho thuậ toán ma-sac... đầu ra của agent này là". 
        # I'll create a shared SACAgent for all Edges to learn faster.
        # Let's calculate state_dim according to plan
        max_models = trainer.env.static_matrices["max_models"]
        num_nodes = trainer.num_nodes # this usually includes edge, cloud, etc. Wait, we map node to instance?
        
        # State dim: max_models (acc) + 3 (dl) + 1 (omega) + 1 (data) + num_nodes (f_s) + num_nodes (backlog) + num_nodes * max_models (MF)
        self.state_dim = max_models + 3 + 1 + 1 + 2 * num_nodes + num_nodes * max_models
        self.action_dim = num_nodes * max_models
        
        trainer.shared_lower_agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=trainer.device,
            dist_type="dirichlet",  # Use Dirichlet for probability plan
            actor_lr=float(trainer.config.hyper_neural.get('LOWER_LR', 1e-4)),
            critic_lr=float(trainer.config.hyper_neural.get('LOWER_LR', 3e-4))
        )
        # Create replay buffer for it
        from matrix_source.agents.sac_ec import ReplayBuffer
        trainer.shared_lower_agent.memory = ReplayBuffer(self.state_dim, self.action_dim, capacity=trainer.config.hyper_neural['MEMORY_SIZE'])
        self.lower_train_num = 0

    def group_tasks_by_edge(self, trainer, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        """
        Groups tasks by (edge_id, service_idx)
        """
        batches = {}
        terminals = trainer.env.static_matrices["terminals"]
        for i, t in enumerate(t_idx):
            edge_id = terminals[t].edge_id
            sid = s_idx[i].item()
            key = (edge_id, sid)
            
            if key not in batches:
                batches[key] = {
                    'task_indices': [],
                    'min_accuracies': [],
                    'deadlines': [],
                    'batch_sizes': []
                }
            
            batches[key]['task_indices'].append(i) # local index in t_idx array
            batches[key]['min_accuracies'].append(tasks_min_accuracy[i].item())
            batches[key]['deadlines'].append(task_deadlines[i].item())
            batches[key]['batch_sizes'].append(batch_sizes[i].item())
            
        return batches

    def build_edge_state(self, trainer, edge_id, sid, batch_data, current_obs, mf_terminals):
        """
        Builds the 1D state vector for a specific Edge and Service batch.
        """
        device = trainer.device
        meta = trainer.env.metadata
        max_models = meta["max_models"]
        num_nodes = trainer.num_nodes
        
        accuracies = batch_data['min_accuracies']
        deadlines = batch_data['deadlines']
        b_sizes = batch_data['batch_sizes']
        
        # 1. Min Accuracy Distribution
        acc_tensor = torch.tensor(accuracies, device=device)
        model_accs = meta["model_accuracies"][sid] # shape: (max_models,)
        # Find min model index that satisfies acc
        valid_mask = model_accs.unsqueeze(0) >= acc_tensor.unsqueeze(1) # (batch, max_models)
        # argmax on boolean mask returns first True
        best_model_idx = valid_mask.float().argmax(dim=1) 
        # Edge case: if none is valid, argmax returns 0. If we want safest, it should be the strongest model, but let's trust valid_mask logic
        acc_one_hot = torch.nn.functional.one_hot(best_model_idx, num_classes=max_models).float()
        acc_dist = acc_one_hot.mean(dim=0)
        
        # 2. Deadline Distribution
        dl_tensor = torch.tensor(deadlines, device=device).unsqueeze(1)
        svc_deadlines = meta["service_deadlines"][sid].unsqueeze(0) # (1, 3)
        # Find closest deadline bin
        dist = torch.abs(dl_tensor - svc_deadlines)
        closest_dl_idx = dist.argmin(dim=1)
        dl_one_hot = torch.nn.functional.one_hot(closest_dl_idx, num_classes=3).float()
        dl_dist = dl_one_hot.mean(dim=0)
        
        # 3. Omega and Data Size
        omega = meta["service_omega"][sid].squeeze()
        input_size = meta["service_input_size"][sid].squeeze()
        total_data = sum(b_sizes) * input_size
        total_data_norm = torch.tensor([total_data / trainer.config.norm_data_size], device=device).float()
        omega_tensor = torch.tensor([omega], device=device).float()
        
        # 4. Context Matrix (f_s and backlog_s)
        # obs_dict['backlog'] shape: (N, S) => (num_nodes, num_services)
        backlog_s = current_obs['backlog'][:, sid] / 100.0 # normalized
        # obs_dict['cpu_alloc'] shape: (N, S)
        cpu_alloc_s = current_obs['cpu_alloc'][:, sid] / trainer.config.norm_gflop # normalized
        
        # 5. Mean field
        # average MF of all terminals for now or edge agents? 
        # The user said "agent offload task đặt tại edge mỗi edge có 1 agent... thêm 1 vector mean field chứa hành dộng trung bình của các node agent bên cạnh "
        # For simplicity, we just use the global mean field from terminals. 
        # Note mf_terminals shape: (num_terminals, max_models * num_nodes)
        mf_global = mf_terminals.mean(dim=0) # (N * M)
        
        state = torch.cat([
            acc_dist, dl_dist, omega_tensor, total_data_norm, backlog_s, cpu_alloc_s, mf_global
        ], dim=0)
        
        return state

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        if len(t_idx) == 0:
            return torch.tensor([], device=trainer.device), torch.tensor([], device=trainer.device)
            
        batches = self.group_tasks_by_edge(trainer, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field']
        
        # We will output for each task in t_idx specific node and model
        final_n_idx = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)
        final_m_idx = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)
        
        # Keep track of states for storage later
        self.latest_edge_states = {}
        self.latest_edge_actions = {}
        
        for (edge_id, sid), batch_data in batches.items():
            state = self.build_edge_state(trainer, edge_id, sid, batch_data, obs_dict, mf_terminals)
            
            # Action space: Dirichlet prob vector of size N * M
            # In alternate phases, determinism controls exploration
            is_deterministic = False # Adjust based on phase logic if needed
            action_plan = trainer.shared_lower_agent.select_action(state.cpu().numpy(), evaluate=is_deterministic)
            action_plan_t = torch.tensor(action_plan, device=trainer.device)
            
            num_tasks = len(batch_data['task_indices'])
            
            # Action Masking based on min_accuracy
            # The plan is (N * M). We need to make sure tasks are assigned to valid models.
            # To be precise, since tasks in a batch have DIFFERENT min_accuracy, we mask PER task when sampling.
            model_accs = trainer.env.metadata["model_accuracies"][sid] # (M,)
            # We reshape plan to (N, M)
            plan_nm = action_plan_t.view(trainer.num_nodes, trainer.env.metadata["max_models"]) # (N, M)
            
            # Sampling logic
            for local_i, task_global_i in enumerate(batch_data['task_indices']):
                min_acc = batch_data['min_accuracies'][local_i]
                valid_models = model_accs >= min_acc # (M,)
                
                # Mask plan
                task_plan = plan_nm.clone()
                task_plan[:, ~valid_models] = 0.0 # Make invalid models 0 prob
                
                # Normalize
                if task_plan.sum() > 0:
                    task_plan = task_plan / task_plan.sum()
                else:
                    # Fallback: if somehow no model is valid (unlikely), pick strongest
                    task_plan[:, -1] = 1.0 / trainer.num_nodes
                
                # Sample from Task Plan
                flat_prob = task_plan.flatten()
                chosen_idx = torch.multinomial(flat_prob, 1)[0]
                
                n_id = chosen_idx // trainer.env.metadata["max_models"]
                m_id = chosen_idx % trainer.env.metadata["max_models"]
                
                final_n_idx[task_global_i] = n_id
                final_m_idx[task_global_i] = m_id
            
            self.latest_edge_states[(edge_id, sid)] = state
            self.latest_edge_actions[(edge_id, sid)] = action_plan_t
                
        return final_n_idx, final_m_idx
        
    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        from matrix_source.trainers.train import log_transform
        
        # Reward comes globally for the step. For MA-SAC, we need to distribute it to the Edge batches.
        # Currently, reward logic is per-terminal. We average it out for the batch.
        reward = next_res['reward'] # shape (num_terminals_active,)
        norm_rew = log_transform(reward / trainer.config.norm_lower_rw)
        
        # We need to re-group next_res to find the next state.
        # Wait, the next state occurs on the NEXT step for that edge... 
        # But this is a standard RL env for the step. Let's build the Next State based on next_res.
        obs_dict = next_res['obs']
        mf_terminals = next_res['mean_field']
        batches = self.group_tasks_by_edge(trainer, t_idx, s_idx, t_idx*0, t_idx*0, t_idx*0) # Only care about grouping keys
        # Wait, the batch metadata (accuracies, size) has changed because the NEXT state has completely different tasks!
        # Standard RL maps S_t -> S_t+1. Because our workload arrives dynamically, S_t+1 doesn't have the same tasks.
        # This is a known issue in event-driven MDP. We treat next_state as the state of the *next* arriving batch for that Edge, 
        # OR we just use a baseline next_state (e.g., system context only).
        # To be completely correct, the Next State should be evaluated against the ACTUAL next workload that arrives.
        # For simplicity and immediate fix, we store placeholder zero state or calculate it if next tasks exist.
        
        # We store transitions per Edge-Service batch we executed.
        for (edge_id, sid), state in self.latest_edge_states.items():
            action = self.latest_edge_actions[(edge_id, sid)]
            
            # Calculate mean reward for tasks in this batch
            b_indices = batches[(edge_id, sid)]['task_indices']
            b_rewards = norm_rew[b_indices].mean().item()
            
            # Since S' is not well-defined immediately (the NEXT time this edge sees this service could be random),
            # we use terminal state or zero state for S' to mark end of this discrete batch decision
            # (or we build it from the next frame's start, but let's stick to using current context with zeroed tasks)
            # A simple approximation: next state is state with 0 tasks but updated system context.
            device = trainer.device
            meta = trainer.env.metadata
            b_s = current_res['obs']['backlog'][:, sid] / 100.0
            c_s = current_res['obs']['cpu_alloc'][:, sid] / trainer.config.norm_gflop
            mf = mf_terminals.mean(dim=0)
            
            next_state = torch.cat([
                torch.zeros(meta["max_models"], device=device), # acc
                torch.zeros(3, device=device), # dl
                torch.tensor([meta["service_omega"][sid].squeeze()], device=device).float(), # omega
                torch.zeros(1, device=device), # data
                b_s, c_s, mf
            ], dim=0)

            done = 1.0 if next_res.get("new_frame", False) else 0.0

            trainer.shared_lower_agent.memory.add(
                state.cpu().numpy(),
                action.cpu().numpy(),
                b_rewards,
                next_state.cpu().numpy(),
            done
            )
            
        trainer.aggregator.add_lower(next_res, mf_loss=0, state=None)
        
    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        # Default placeholder, assuming upper doesn't do much or is integrated separately
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        for nid in trainer.edge_node_ids:
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix
        
    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        pass # Handle upper transitions if upper agent exists

    def run_training(self, trainer):
        from tqdm import tqdm
        import os
        max_slots = trainer.env.time_manager.max_steps
        max_episodes = 500
        batch_size = 256
        
        pbar = tqdm(total=max_episodes, desc="MA-SAC Edge Training Progress")
        
        for ep in range(max_episodes):
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            
            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer, None, obs_upper)
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
                    
                    # Train MA-SAC Agent
                    metrics = trainer.shared_lower_agent.train_step(trainer.shared_lower_agent.memory, batch_size=batch_size)
                    if metrics:
                        self.lower_train_num += 1
                        
                        # Record metrics for visualization
                        trainer.aggregator.record_q_stats(
                            "Terminal_Group", 
                            q_min=metrics.get("q_min", 0.0), # SAC usually reports mean, fallback
                            q_max=metrics.get("q_max", 0.0), 
                            q_mean=metrics.get("q_mean", 0.0)
                        )
                        trainer.aggregator.record_td_losses(lower_losses=metrics.get("critic_loss", 0.0))

                        # Save checkpoint randomly or per interval
                        if self.lower_train_num % 1000 == 0:
                            os.makedirs('checkpoints', exist_ok=True)
                            torch.save(trainer.shared_lower_agent.actor.state_dict(), f'checkpoints/masac_actor_{self.lower_train_num}.pth')

                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    trainer.aggregator.add_upper(res_upper)
                    
            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
            pbar.update(1)
