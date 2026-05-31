from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch

from matrix_source.configs.abstract_config import (
    AbstractTrainingConfig,
)
from matrix_source.utils.config_updater import (
    apply_config_to_agent, 
    start_convergence, 
    create_entropy_scheduler, create_lr_schedulers
)


class AbstractStrategy(ABC):
    """
    Template strategy cho MỌI kiến trúc.

    Training loop (template method pattern):

        for cycle in range(1, MAX_CYCLES + 1):
            phase = get_phase(cycle)
            zeta_u, zeta_l = apply_phase_config(cycle, phase)

            for slot in range(max_slots):
                # Upper act
                # Lower act
                # Train lower
                # Train upper

            post_episode_hook()  # SCAFFOLD sync, etc.
    """

    def __init__(self, config_class: type):
        """
        Args:
            config_class: Class kế thừa AbstractTrainingConfig
        """
        self.config: AbstractTrainingConfig = config_class
        self.is_evaluating: bool = False
        self.upper_mf_ema: Optional[torch.Tensor] = None
        self.mf_ema_alpha: float = 0.3

    # ════════════════════════════════════════════════════
    # ABSTRACT METHODS — Bắt buộc implement
    # ════════════════════════════════════════════════════

    @abstractmethod
    def initialize_agents(self, trainer) -> None:
        """
        Khởi tạo upper_agent và lower_agent.

        Bắt buộc:
          trainer.shared_upper_agent = SomePPOAgent(...)
          trainer.shared_lower_agent = SomeLowerAgent(...)

        Sau khi init, gọi:
          self._setup_agent_infra(agent, entropy_config, config)
        """
        pass

    @abstractmethod
    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        """Returns: (act_matrix, log_probs, values)"""
        pass

    @abstractmethod
    def get_lower_actions(self, trainer, *args, **kwargs):
        """Returns: varies by architecture"""
        pass

    @abstractmethod
    def store_upper_transitions(self, trainer, *args, **kwargs):
        """Store upper-level transitions."""
        pass

    @abstractmethod
    def store_lower_transitions(self, trainer, *args, **kwargs):
        """Store lower-level transitions."""
        pass

    @abstractmethod
    def train_lower(self, trainer, zeta: float) -> Optional[float]:
        """Train lower agent. Returns loss or None."""
        pass

    @abstractmethod
    def train_upper(self, trainer, zeta: float) -> Optional[float]:
        """Train upper agent. Returns loss or None."""
        pass

    # ════════════════════════════════════════════════════
    # TEMPLATE METHODS — Override nếu cần
    # ════════════════════════════════════════════════════

    def build_upper_state(self, trainer, obs_upper):
        """Build upper state từ observation."""
        acts = obs_upper['actions']
        phi = obs_upper['phi_prob']
        return torch.cat([acts, phi], dim=1)

    def on_cycle_start(self, trainer, cycle: int, phase: str):
        """Hook: gọi đầu mỗi cycle. Override nếu cần."""
        pass

    def on_cycle_end(self, trainer, cycle: int, phase: str):
        """Hook: gọi cuối mỗi cycle. Override cho SCAFFOLD sync."""
        pass

    def on_phase_change(self, trainer, old_phase: str, new_phase: str):
        """Hook: gọi khi phase thay đổi. Override nếu cần."""
        pass

    # ════════════════════════════════════════════════════
    # UNIVERSAL SETUP — Gọi trong initialize_agents()
    # ════════════════════════════════════════════════════

    def _setup_agent_infra(self, agent, entropy_cfg, config,
                        est_train_steps=2000,
                        lr_min_overrides=None,
                        converge_steps=500):
        """Setup: entropy scheduler + LR schedulers. No target networks."""
        from matrix_source.utils.config_updater import (
            create_entropy_scheduler, create_lr_schedulers
        )

        # 1. Entropy scheduler
        agent.entropy_sched = create_entropy_scheduler(
            entropy_cfg, total_train_steps=est_train_steps)

        # 2. LR schedulers
        agent.lr_schedulers = create_lr_schedulers(
            agent,
            lr_min_actor=1e-6,
            lr_min_critic=5e-6,
            total_steps=converge_steps
        )

    # ════════════════════════════════════════════════════
    # UNIVERSAL PHASE CONFIG — Gọi mỗi cycle
    # ════════════════════════════════════════════════════

    def apply_phase_config(self, cycle: int) -> Tuple[float, float]:
        """
        Apply phase config cho CẢ 2 agents.
        Returns: (zeta_lower, zeta_upper)

        Override _apply_lower_phase_config() nếu lower cần custom apply.
        """
        phase = self.config.get_phase(cycle)
        phase_bounds = self.config.PHASE_BOUNDS[phase]
        is_converge = (cycle >= self.config.CONVERGE_START)

        # Chuyển sang CONVERGE
        if cycle == self.config.CONVERGE_START:
            self._start_convergence()

        # Apply upper
        temp_u, zeta_u = apply_config_to_agent(
            self._upper_agent,
            self.config.UPPER_PHASES[phase],
            cycle, phase_bounds, is_converge
        )

        # Apply lower (có thể override)
        temp_l, zeta_l = self._apply_lower_phase_config(
            cycle, phase, phase_bounds, is_converge
        )

        return zeta_l, zeta_u

    def _apply_lower_phase_config(self, cycle, phase, phase_bounds, is_converge):
        """
        Apply config cho lower agent. Override cho kiến trúc custom.
        Default: dùng apply_config_to_agent() chuẩn.
        """
        return apply_config_to_agent(
            self._lower_agent,
            self.config.LOWER_PHASES[phase],
            cycle, phase_bounds, is_converge
        )

    def _start_convergence(self):
        """Kích hoạt LR cosine annealing cho cả 2 agents."""
        if hasattr(self, '_upper_agent') and self._upper_agent.lr_schedulers:
            start_convergence(self._upper_agent.lr_schedulers)
        if hasattr(self, '_lower_agent') and self._lower_agent.lr_schedulers:
            start_convergence(self._lower_agent.lr_schedulers)

    # ════════════════════════════════════════════════════
    # MAIN TRAINING LOOP — Template method
    # ════════════════════════════════════════════════════

    def run_training(self, trainer):
        """
        Template training loop.
        Override các hook methods để customize.
        """
        max_slots = trainer.env.time_manager.max_steps

        for cycle in range(1, self.config.MAX_CYCLES + 1):
            phase = self.config.get_phase(cycle)

            # Apply configs
            zeta_l, zeta_u = self.apply_phase_config(cycle)

            # Hook: cycle start
            self.on_cycle_start(trainer, cycle, phase)

            # Episode
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            current_upper_state = self.build_upper_state(trainer, obs_upper)

            for slot in range(max_slots):
                # Upper act
                if trainer.env.time_manager.is_new_frame():
                    u_acts, u_lp, u_v = self.get_upper_actions(
                        trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts)

                # Lower act + train
                lower_result = self._process_lower_slot(
                    trainer, obs, slot, zeta_l)

                # Upper transitions + train
                if trainer.env.time_manager.is_new_frame():
                    self._process_upper_frame(
                        trainer, obs_upper, current_upper_state,
                        u_acts, u_lp, u_v, slot, zeta_u)
                    current_upper_state = self._build_next_upper_state(trainer)
                    obs_upper = self._get_next_obs_upper(trainer)

            # Hook: cycle end
            self.on_cycle_end(trainer, cycle, phase)

            # Log
            self._log_cycle(trainer, cycle, phase, zeta_l, zeta_u)

    def _process_lower_slot(self, trainer, obs, slot, zeta_l):
        """Xử lý 1 slot cho lower. Override cho kiến trúc cụ thể."""
        t_idx, s_idx, batch_sizes, min_acc, deadlines = \
            trainer.workload_gen.generate_step()

        if len(t_idx) > 0:
            # Get actions
            lower_out = self.get_lower_actions(
                trainer, obs['lower'], t_idx, s_idx,
                min_acc, deadlines, batch_sizes)

            # Store + train
            self.store_lower_transitions(trainer, *lower_out)
            loss = self.train_lower(trainer, zeta_l)
            if loss is not None:
                trainer.total_lower_steps += 1
                trainer.aggregator.record_td_losses(lower_losses=loss)
        else:
            trainer.env.time_manager.tick()

    def _process_upper_frame(self, trainer, obs_upper, current_upper_state,
                              u_acts, u_lp, u_v, slot, zeta_u):
        """Xử lý 1 frame cho upper."""
        res_upper = trainer.env.collect_upper_metrics()
        next_upper_state = self.build_upper_state(trainer, res_upper)
        is_done = (slot == trainer.env.time_manager.max_steps - 1)

        self.store_upper_transitions(
            trainer, current_upper_state, next_upper_state,
            obs_upper, res_upper, u_acts, u_lp, u_v, is_done)

        loss = self.train_upper(trainer, zeta_u)
        if loss is not None:
            trainer.total_upper_steps += 1
            trainer.aggregator.record_td_losses(upper_losses=loss)

    def _build_next_upper_state(self, trainer):
        res = trainer.env.collect_upper_metrics()
        return self.build_upper_state(trainer, res)

    def _get_next_obs_upper(self, trainer):
        return trainer.env.collect_upper_metrics()

    # ════════════════════════════════════════════════════
    # LOGGING
    # ════════════════════════════════════════════════════

    def _log_cycle(self, trainer, cycle, phase, zeta_l, zeta_u):
        ent_l = self._lower_agent.entropy_coef
        ent_u = self._upper_agent.entropy_coef
        print(f"[Cycle {cycle:2d}/{self.config.MAX_CYCLES}] "
              f"Phase={phase:8s} | "
              f"ζ_L={zeta_l:.3f} ζ_U={zeta_u:.3f} | "
              f"EntL={ent_l:.5f} EntU={ent_u:.5f}")

    # ════════════════════════════════════════════════════
    # EVALUATION
    # ════════════════════════════════════════════════════

    @abstractmethod
    def run_evaluation(self, trainer, num_episodes=5):
        """Post-training evaluation."""
        pass
