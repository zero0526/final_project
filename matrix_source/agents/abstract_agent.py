from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch
import torch.optim as optim


class HasOptimizers:
    def get_optimizers(self) -> Dict[str, optim.Optimizer]:
        result = {}
        name_map = {
            'optimizer_actor': 'actor',
            'optimizer_critic': 'critic',
            'optimizer_proposal': 'proposal',
            'optimizer_refine': 'refine',
            'bone_optimizer': 'backbone',
            'head_optimizer': 'head',
            'mf_optimizer': 'mf',
        }
        for attr, name in name_map.items():
            if hasattr(self, attr):
                result[name] = getattr(self, attr)
        return result

    def set_lr_factor(self, factor: float):
        for opt in self.get_optimizers().values():
            for pg in opt.param_groups:
                pg['lr'] *= factor


class AbstractAgent(HasOptimizers, ABC):

    def __init__(self):
        self.device: torch.device = torch.device('cpu')
        self.num_instances: int = 1
        self.eps_clip: float = 0.2
        self.k_epochs: int = 3

        self.entropy_coef: float = 0.01
        self.entropy_sched: Any = None
        self.lr_schedulers: Dict = {}

    @abstractmethod
    def learn(self, zeta: float = 1.0, **kwargs) -> Optional[float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def choose_action_batch(self, *args, **kwargs):
        raise NotImplementedError

    def store_transition_train_mf_batch(self, *args, **kwargs):
        raise NotImplementedError

    def update_entropy(self, avg_reward: float = None) -> float:
        if self.entropy_sched is not None:
            self.entropy_coef = self.entropy_sched.step(avg_reward)
        return self.entropy_coef

    def step_lr_schedulers(self):
        from matrix_source.utils.config_updater import step_all_schedulers
        step_all_schedulers(self.lr_schedulers)

    def get_entropy_phase(self) -> str:
        if self.entropy_sched is not None:
            return self.entropy_sched.get_phase_name()
        return 'UNKNOWN'

    # REMOVED: soft_update_targets() — not needed without target networks
