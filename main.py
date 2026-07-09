from matrix_source.trainers.train import Trainer
from matrix_source.trainers.ppo_stategy import PPOStrategy
from matrix_source.trainers.d3qn_strategy import D3QNStrategy
from matrix_source.trainers.d3qn_scaffold_strategy_v2 import D3QNScaffoldStrategy
from matrix_source.trainers.semi_distribute_task import GRUPPOSCAFFOLDREPStrategy
# from matrix_source.trainers.ppo_stategy_v2 import PPOSCAFFOLDREPStrategy
from matrix_source.trainers.residual_routing_ppo import ResidualRoutingPPOStrategy
from matrix_source.configs.configs import cfg
cfg.hyper_neural["NUM_LOWER_AGENTS"]= 1000

cfg.hyper_neural["SLOT_DURATION"]= 1.0
cfg.lypa_coef= 3e-3

cfg.max_queue_size = 5000
cfg.admm_max_iter = 50
if __name__ == '__main__':
    strategy = ResidualRoutingPPOStrategy()
    strategy.max_cycles = 900
    strategy.proposal_only_cycles= 950
    trainer = Trainer(strategy=strategy)
    trainer.train()