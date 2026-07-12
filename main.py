from matrix_source.trainers.train import Trainer
from matrix_source.trainers.ppo_stategy import PPOStrategy
from matrix_source.trainers.d3qn_strategy import D3QNStrategy
from matrix_source.trainers.d3qn_scaffold_strategy_v2 import D3QNScaffoldStrategy
from matrix_source.trainers.semi_distribute_task import GRUPPOSCAFFOLDREPStrategy
# from matrix_source.trainers.ppo_stategy_v2 import PPOSCAFFOLDREPStrategy
from matrix_source.trainers.residual_routing_ppo import ResidualRoutingPPOStrategy
from matrix_source.configs.configs import cfg

if __name__ == '__main__':
    num_terminals= 20
    cfg.hyper_neural["NUM_LOWER_AGENTS"] = num_terminals
    strategy = ResidualRoutingPPOStrategy()
    trainer = Trainer(strategy=strategy)
    trainer.config_scenario(num_terminals=num_terminals)
    trainer.train(max_cycles_finetune=1000)
