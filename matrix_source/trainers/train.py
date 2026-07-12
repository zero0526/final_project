from matrix_source.envs.matrix_env import MatrixSixGEnvironment
from matrix_source.envs.workload_generator import MatrixWorkloadGenerator
from matrix_source.configs.configs import cfg
from matrix_source.visualize.aggregator import MetricsAggregator
from matrix_source.trainers.ppo_stategy import PPOStrategy

class Trainer:
    def __init__(self, strategy=None):
        self.config = cfg
        self.device = cfg.hyper_neural.get('DEVICE', cfg.device)
        
        # 1. Initialize Environment & Workload
        self.env = MatrixSixGEnvironment(config=cfg, device=self.device)
        self.workload_gen = MatrixWorkloadGenerator(cfg, self.env.metadata, device=self.device)

        self.num_services = self.env.engine.num_services
        self.num_nodes = self.env.engine.num_nodes
        self.num_terminals = self.workload_gen.num_terminals
        self.max_models = self.env.metadata.get("max_models", 5)

        # State Dims
        self.upper_state_dim = (self.num_services * 2)
        self.lower_state_dim = 4 + (self.num_nodes * 2) 

        self.upper_action_dim = self.num_services
        self.upper_u_action_dim = 1 << self.num_services
        self.lower_action_dim = self.num_nodes + self.max_models
        self.lower_u_action_dim = self.num_nodes * self.max_models

        # --- Hyperparams ---
        self.min_epsilon = cfg.hyper_neural.get("EPSILON", 0.05)
        self.epsilon_decay = cfg.hyper_neural.get("EPSILON_DECAY", 0.9985)
        self.eps_upper = 1.0
        self.eps_lower = 1.0
        self.lower_epsilons = {tid: 1.0 for tid in range(self.num_terminals)}
        self.zeta_initial_upper = cfg.hyper_neural.get("ZETA", 1.0)
        self.zeta_initial_lower = cfg.hyper_neural.get("ZETA", 1.0)
        self.zeta_max_upper = cfg.hyper_neural.get("ZETA_MAX", 20.0)
        self.zeta_max_lower = cfg.hyper_neural.get("ZETA_MAX", 20.0)

        self.zeta_upper = self.zeta_initial_upper
        self.zeta_lower = self.zeta_initial_lower
        self.epsilon_start = 1.0
        self.annealing = cfg.hyper_neural.get("ANNEALING_LENGTH", 5000)
        # --- Lower-level zeta exploration schedule (independent from upper) ---
        # Lower zeta stays frozen during warmup, then increases slowly per episode
        self.zeta_lower_warmup = int(cfg.hyper_neural.get("ZETA_LOWER_WARMUP", 200))   # episodes to keep zeta_lower = initial
        self.zeta_lower_step  = float(cfg.hyper_neural.get("ZETA_LOWER_STEP", 0.005))  # additive increment per episode after warmup
        self.zeta_lower_max   = float(cfg.hyper_neural.get("ZETA_LOWER_MAX", 5.0))     # separate cap for lower (keep < upper cap)

        # Training control variables
        self.total_lower_steps = 0
        self.total_upper_steps = 0
        self.lower_stable_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][0]*10
        self.lower_start_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][1]
        
        self.aggregator = MetricsAggregator()
        self.shared_upper_agent = None
        self.shared_lower_agent = None

        self.edge_ids = self.env.static_matrices["edge_ids"]
        self.edge_node_ids = [nid for nid in range(self.num_nodes)
                              if nid not in self.env.static_matrices.get("cloud_ids", [])]
        self.node_to_instance = {nid: i for i, nid in enumerate(self.edge_node_ids)}
        self.max_epochs= 3000
        self.num_edge_agents = len(self.edge_node_ids)
        # 2. Strategy Injection
        self.strategy = strategy if strategy is not None else PPOStrategy()
        self.aggregator.name = self.strategy.__class__.__name__
        self.strategy.initialize_agents(self)

    def train(self, resume_from=None, save_every=None, save_dir="checkpoints", **kwargs):
        """
        Runs the injected strategy's training loop with support for checkpoint loading and saving.
        """
        if save_every is None:
            strat_name = self.strategy.__class__.__name__
            if "PPO" in strat_name:
                save_every = 1
            else:
                save_every = 500

        self.strategy.run_training(
            self,
            resume_from=resume_from,
            save_every=save_every,
            save_dir=save_dir,
            **kwargs
        )

    def evaluate(self, checkpoint_dir, num_eval_eps=10, label=None, plot_dir="eval_results", compare_results=None, **kwargs):
        """
        Runs greedy evaluation episodes using the model weights loaded from checkpoint_dir,
        computes the standard performance metrics, and plots the results.
        """
        if label is None:
            label = self.strategy.__class__.__name__.replace("Strategy", "")
        
        return self.strategy.evaluate(
            self,
            checkpoint_dir=checkpoint_dir,
            num_eval_eps=num_eval_eps,
            label=label,
            plot_dir=plot_dir,
            compare_results=compare_results,
            **kwargs
        )

    def update_rates(self, ep):
        # 1. Update Epsilons using exponential decay: eps = eps_end + (eps_start - eps_end) * exp(-t / tau)
        import math
        self.eps_upper = self.min_epsilon + (self.epsilon_start - self.min_epsilon) * \
                         math.exp(-self.total_upper_steps / 10 / self.annealing)

        self.eps_lower = self.min_epsilon + (self.epsilon_start - self.min_epsilon) * \
                         math.exp(-self.total_lower_steps / 100 / self.annealing)

        # zeta_upper: bắt đầu từ zeta_initial (=1.0), tăng dần lên zeta_max khi có nhiều upper steps
        # Dùng (1 - exp) để tăng từ 0→1 thay vì exp giảm từ 1→0
        self.zeta_upper = self.zeta_initial_upper + (self.zeta_max_upper - self.zeta_initial_upper) * \
                         (1.0 - math.exp(-self.total_upper_steps / 10 / self.annealing))

        # zeta_lower: giữ nguyên initial trong warmup eps, sau đó tăng từng bước nhỏ
        if ep < self.zeta_lower_warmup:
            self.zeta_lower = self.zeta_initial_lower
        else:
            self.zeta_lower = min(
                self.zeta_initial_lower + self.zeta_lower_step * (ep - self.zeta_lower_warmup),
                self.zeta_lower_max
            )
    def config_scenario(self, num_terminals=20):
        self.strategy.max_cycles = 950
        self.strategy.proposal_only_cycles= 950
        if num_terminals == 20:
            cfg.hyper_neural["NUM_LOWER_AGENTS"]= 20
            cfg.norm_upper_rw= 1_000_000_0.0
            cfg.norm_lower_rw= 1_000_000_0.0
        elif num_terminals == 40:
            cfg.hyper_neural["NUM_LOWER_AGENTS"]= 40
            cfg.norm_upper_rw= 5_000_000_0.0
            cfg.norm_lower_rw= 5_000_000_0.0
        elif num_terminals == 60:
            cfg.hyper_neural["NUM_LOWER_AGENTS"]= 60
            cfg.norm_upper_rw= 10_000_000_0.0
            cfg.norm_lower_rw= 10_000_000_0.0
            cfg.hyper_neural["SLOT_DURATION"]= 1.0
def log_transform(reward: float) -> float:
    return reward

if __name__ == "__main__":
    Trainer().train()