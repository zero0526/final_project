from matrix_source.trainers.train import Trainer
from matrix_source.trainers.ppo_stategy import PPOStrategy
from matrix_source.trainers.d3qn_strategy import D3QNStrategy
from matrix_source.trainers.d3qn_scaffold_strategy_v2 import D3QNScaffoldStrategy
from matrix_source.trainers.semi_distribute_task import GRUPPOSCAFFOLDREPStrategy
# from matrix_source.trainers.ppo_stategy_v2 import PPOSCAFFOLDREPStrategy
from matrix_source.trainers.residual_routing_ppo import ResidualRoutingPPOStrategy
from matrix_source.configs.configs import cfg

if __name__ == '__main__':
    strategy = ResidualRoutingPPOStrategy()
    trainer = Trainer(strategy=strategy)
    trainer.config_scenario(num_terminals=20)
    trainer.train()