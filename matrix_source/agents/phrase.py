from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
from torch.distributions import Categorical


@dataclass
class PhaseParameters:
    """Các tham số mà Phase cung cấp cho Agent"""
    alpha: float = 1.0  # Weight cho proposal
    beta: float = 0.0  # Weight cho refine
    entropy_coef_proposal: float = 0.05
    entropy_coef_refine: float = 0.0
    train_proposal: bool = True
    train_refine: bool = False
    freeze_proposal: bool = False
    freeze_refine: bool = True
    grad_clip_proposal: float = 1.0
    grad_clip_refine: float = 1.0
    lr_proposal: Optional[float] = None  # None = dùng default
    lr_refine: Optional[float] = None


class BasePhase(ABC):
    """Abstract base class cho tất cả phases"""

    def __init__(self, name: str, default_hp: Dict[str, Any]):
        self.name = name
        self.hp = default_hp.copy()  # Hyperparameters có thể cập nhật
        self.step_counter = 0  # Số steps trong phase này
        self._history = []  # Track HP changes

    # ══════════════════════════════════════════════
    # ABSTRACT METHODS (Mỗi phase phải implement)
    # ══════════════════════════════════════════════

    @abstractmethod
    def get_parameters(self) -> PhaseParameters:
        """Trả về tham số cho Agent (alpha, beta, entropy, etc.)"""
        pass

    @abstractmethod
    def compute_losses(self, agent, batch_data) -> Dict[str, torch.Tensor]:
        """Tính toán losses cho phase này"""
        pass

    @abstractmethod
    def should_transition(self, metrics: Dict[str, float]) -> Optional[str]:
        """
        Kiểm tra điều kiện chuyển phase.
        Returns: name của phase tiếp theo, hoặc None nếu chưa chuyển
        """
        pass

    # ══════════════════════════════════════════════
    # LIFECYCLE METHODS
    # ══════════════════════════════════════════════

    def on_enter(self, agent):
        """Called khi vào phase này"""
        self.step_counter = 0
        print(f"\n{'=' * 60}")
        print(f"ENTERING PHASE: {self.name}")
        print(f"{'=' * 60}\n")

    def on_exit(self, agent):
        """Called khi rời phase này"""
        print(f"\n{'=' * 60}")
        print(f"EXITING PHASE: {self.name} after {self.step_counter} steps")
        print(f"{'=' * 60}\n")

    def step(self):
        """Called mỗi learning step"""
        self.step_counter += 1

    # ══════════════════════════════════════════════
    # HYPERPARAMETER MANAGEMENT
    # ══════════════════════════════════════════════

    def update_hp(self, key: str, value: Any):
        """Cập nhật hyperparameter (với tracking)"""
        old_value = self.hp.get(key)
        self.hp[key] = value
        self._history.append({
            'step': self.step_counter,
            'key': key,
            'old': old_value,
            'new': value,
        })

    def get_hp(self, key: str, default=None):
        """Lấy hyperparameter"""
        return self.hp.get(key, default)

    # ══════════════════════════════════════════════
    # SERIALIZATION
    # ══════════════════════════════════════════════

    def save_state(self) -> Dict[str, Any]:
        """Lưu state của phase (hp + history)"""
        return {
            'name': self.name,
            'hp': self.hp.copy(),
            'step_counter': self.step_counter,
            'history': self._history.copy(),
        }

    def load_state(self, state: Dict[str, Any]):
        """Load state của phase"""
        assert state['name'] == self.name, \
            f"Phase name mismatch: {state['name']} != {self.name}"
        self.hp = state['hp'].copy()
        self.step_counter = state['step_counter']
        self._history = state['history'].copy()


class ProposalOnlyPhase(BasePhase):
    """
    Phase 1: Chỉ train Proposal
    - Refine bị đóng băng (beta = 0)
    - Entropy decay từ 0.05 → 0.001
    """

    def __init__(self, default_hp: Dict[str, Any] = None):
        # Merge default_hp với custom defaults
        merged_hp = {
            'entropy_coef_start': 0.05,
            'entropy_coef_end': 0.001,
            'entropy_decay_rate': 0.995,
            'min_steps': 5000,
            'reward_stable_window': 10,  # Tăng từ 30 → 100
            'reward_stable_threshold': 0.1,
            'lr_proposal': 3e-4,
            'lr_critic': 1e-3,
        }
        if default_hp:
            merged_hp.update(default_hp)

        # Gọi parent TRƯỚC
        super().__init__(name="ProposalOnly", default_hp=merged_hp)

        # Internal state
        self._current_entropy_coef = self.hp['entropy_coef_start']
        self._reward_history = []

    # LIFECYCLE METHODS
    def on_enter(self, agent):
        """
        Called khi vào phase này.
        Reset internal state và prepare cho phase mới.
        """
        # Reset entropy schedule
        self._current_entropy_coef = self.hp['entropy_coef_start']

        # Reset reward tracking
        self._reward_history = []

        # Call parent to reset step_counter và print message
        super().on_enter(agent)

        print(f"  Initial entropy_coef: {self._current_entropy_coef:.6f}")
        print(f"  Target: {self.hp['entropy_coef_start']:.6f} → {self.hp['entropy_coef_end']:.6f}")
        print(f"  Decay rate: {self.hp['entropy_decay_rate']}")

    def on_exit(self, agent):
        """
        Called khi rời phase này.
        Log summary và cleanup.
        """
        # Log phase summary
        print(f"\n  Phase Summary:")
        print(f"    Steps in phase: {self.step_counter}")
        print(f"    Final entropy_coef: {self._current_entropy_coef:.6f}")
        print(f"    Reward history length: {len(self._reward_history)}")

        if self._reward_history:
            recent = self._reward_history[-min(100, len(self._reward_history)):]
            mean_r = sum(recent) / len(recent)
            print(f"    Avg reward (last {len(recent)}): {mean_r:.3f}")

        # Call parent to print exit message
        super().on_exit(agent)

    def step(self):
        """
        Called mỗi learning step.
        Decay entropy coefficient.
        """
        # Decay entropy
        self._current_entropy_coef = max(
            self._current_entropy_coef * self.hp['entropy_decay_rate'],
            self.hp['entropy_coef_end']
        )

        # Call parent to increment step_counter
        super().step()

    # PHASE PARAMETERS
    def get_parameters(self) -> PhaseParameters:
        """Proposal only: alpha=1, beta=0"""
        return PhaseParameters(
            alpha=1.0,
            beta=0.0,
            entropy_coef_proposal=self._current_entropy_coef,
            entropy_coef_refine=0.0,
            train_proposal=True,
            train_refine=False,
            freeze_proposal=False,
            freeze_refine=True,
            lr_proposal=self.hp['lr_proposal'],
            lr_refine=None,
        )

    def compute_losses(self, agent, batch_data) -> Dict[str, torch.Tensor]:
        """Tính loss cho Proposal only"""
        # Unpack batch
        prop_logits = batch_data['prop_logits']
        act_cat = batch_data['act_cat']
        advantages = batch_data['b_adv']
        old_log_probs = batch_data['b_old_lp']
        masks_exp = batch_data['masks_exp']
        batch_idx = batch_data['batch_idx']
        B_sub = batch_data['B_sub']
        # b_t_lens = batch_data['b_t_lens'] # Không còn cần dùng đến nữa

        # Final logits = alpha * proposal (beta = 0)
        final_logits = agent.mask_and_sanitize(prop_logits, masks_exp)
        dist = Categorical(logits=final_logits)

        # ══════════════════════════════════════════
        # 1. PPO Loss (Dùng nguyên Tổng Sum, KHÔNG CHIA)
        # ══════════════════════════════════════════
        sum_lp = torch.zeros(B_sub, device=agent.device).scatter_add_(
            0, batch_idx, dist.log_prob(act_cat)
        )

        # Lấy trực tiếp sum_lp trừ đi old_log_probs (cũng đang ở dạng tổng)
        ratio = torch.exp(sum_lp - old_log_probs)

        ppo_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages
        ).mean()

        # ══════════════════════════════════════════
        # 2. Entropy Bonus (Dùng nguyên Tổng Sum)
        # ══════════════════════════════════════════
        # sum_ent là tổng entropy của tất cả các task thuộc về 1 agent
        sum_ent = torch.zeros(B_sub, device=agent.device).scatter_add_(
            0, batch_idx, dist.entropy()
        )

        # Chỉ lấy trung bình (mean) theo Batch Size (B_sub)
        entropy_loss = -self._current_entropy_coef * sum_ent.mean()

        return {
            'loss_proposal': ppo_loss + entropy_loss,
            'loss_refine': None,  # Không có refine loss
            'metrics': {
                'entropy_coef': self._current_entropy_coef,
                'entropy': sum_ent.mean().item(),  # Log tổng entropy trung bình của agents
                'ppo_loss': ppo_loss.item(),
            }
        }
    # TRANSITION LOGIC
    def should_transition(self, metrics: Dict[str, float]) -> Optional[str]:
        """
        Kiểm tra điều kiện chuyển sang ProposalFree.

        Điều kiện:
        1. Đã train đủ min_steps
        2. Reward ổn định (std < 10% mean) trong reward_stable_window
        3. Ổn định liên tục trong 100 steps
        """
        # Track reward
        if 'avg_reward' in metrics:
            self._reward_history.append(metrics['avg_reward'])

        # Điều kiện 1: Đã train đủ steps
        if self.step_counter < self.hp['min_steps']:
            return None

        # Điều kiện 2: Reward stable
        window = self.hp['reward_stable_window']
        if len(self._reward_history) >= window:
            recent = self._reward_history[-window:]
            mean_r = sum(recent) / len(recent)
            std_r = (sum((r - mean_r) ** 2 for r in recent) / len(recent)) ** 0.5

            stable_thresh = self.hp['reward_stable_threshold']

            if std_r < abs(mean_r) * stable_thresh:
                return "ProposalFree"

        return None

    # SERIALIZATION (Override để save internal state)
    def save_state(self) -> Dict[str, Any]:
        """Save state bao gồm internal state"""
        state = super().save_state()
        state.update({
            'current_entropy_coef': self._current_entropy_coef,
            'reward_history': self._reward_history.copy(),
        })
        return state

    def load_state(self, state: Dict[str, Any]):
        """Load state bao gồm internal state"""
        super().load_state(state)
        self._current_entropy_coef = state.get(
            'current_entropy_coef',
            self.hp['entropy_coef_start']
        )
        self._reward_history = state.get('reward_history', []).copy()

class ProposalFreePhase(BasePhase):
    """
    Phase 2: Chỉ train Refine để cải thiện kết quả đã có của Proposal

    Chiến lược:
    - Proposal FROZEN hoàn toàn (đã hội tụ từ Phase 1)
    - Refine explore MẠNH (entropy cao, LR cao)
    - Chấp nhận phân phối final thay đổi để tìm giải pháp tốt hơn
    - KHÔNG có KL penalty (không cần tôn trọng proposal)

    Mục tiêu: Check xem Refine có khả năng cải thiện kết quả không
    """

    def __init__(self, default_hp: Dict[str, Any] = None):
        # Merge default_hp với custom defaults
        merged_hp = {
            # Fusion weights
            'alpha': 1.0,  # Weight cho proposal (frozen)
            'beta': 1.0,  # Weight cho refine (active)

            # Entropy schedule cho Refine (explore mạnh → giảm dần)
            'entropy_coef_start': 0.1,  # Cao để explore
            'entropy_coef_end': 0.01,  # Giảm khi đã học
            'entropy_decay_rate': 0.995,  # Decay chậm

            # Learning rates
            'lr_proposal': 0.0,  # FROZEN - không update proposal
            'lr_refine': 1e-3,  # CAO - refine cần học nhanh
            'lr_critic': 1e-3,

            # Grad clipping
            'grad_clip_proposal': 0.0,  # Không cần (không train)
            'grad_clip_refine': 1.0,  # Cao hơn để cho phép update mạnh
        }
        if default_hp:
            merged_hp.update(default_hp)

        # Gọi parent TRƯỚC
        super().__init__(name="ProposalFree", default_hp=merged_hp)

        # Internal state để tracking
        self._delta_norm_history = []
        self._contribution_ratio_history = []
        self._reward_history = []

        # Entropy coefficient (sẽ decay trong step())
        self._current_entropy_coef = self.hp['entropy_coef_start']

    # ══════════════════════════════════════════════════════════
    # LIFECYCLE METHODS
    # ══════════════════════════════════════════════════════════

    def on_enter(self, agent):
        """
        Called khi vào phase này.
        Reset tracking và prepare cho Refine exploration.
        """
        # Reset tracking
        self._delta_norm_history = []
        self._contribution_ratio_history = []
        self._reward_history = []

        # Reset entropy
        self._current_entropy_coef = self.hp['entropy_coef_start']

        # Call parent
        super().on_enter(agent)

        print(f"\n  ═══════════════════════════════════════════")
        print(f"  Phase 2: Refine Exploration")
        print(f"  ═══════════════════════════════════════════")
        print(f"  Strategy: Proposal FROZEN, Refine EXPLORE")
        print(f"  Fusion weights: alpha={self.hp['alpha']}, beta={self.hp['beta']}")
        print(f"  Entropy schedule: {self.hp['entropy_coef_start']:.3f} → "
              f"{self.hp['entropy_coef_end']:.3f} (decay={self.hp['entropy_decay_rate']})")
        print(f"  Learning rates: proposal={self.hp['lr_proposal']} (FROZEN), "
              f"refine={self.hp['lr_refine']} (HIGH)")
        print(f"  ═══════════════════════════════════════════\n")

    def on_exit(self, agent):
        """
        Called khi rời phase này.
        Log summary để đánh giá Refine performance.
        """
        print(f"\n  ═══════════════════════════════════════════")
        print(f"  Phase 2 Summary")
        print(f"  ═══════════════════════════════════════════")
        print(f"    Steps in phase: {self.step_counter}")
        print(f"    Final entropy_coef: {self._current_entropy_coef:.4f}")

        if self._reward_history:
            recent = self._reward_history[-min(100, len(self._reward_history)):]
            mean_r = sum(recent) / len(recent)
            print(f"    Avg reward (last {len(recent)}): {mean_r:.3f}")

        if self._delta_norm_history:
            recent_delta = self._delta_norm_history[-min(100, len(self._delta_norm_history)):]
            mean_delta = sum(recent_delta) / len(recent_delta)
            print(f"    Avg ||Δ|| (last {len(recent_delta)}): {mean_delta:.3f}")

        if self._contribution_ratio_history:
            recent_contrib = self._contribution_ratio_history[-min(100, len(self._contribution_ratio_history)):]
            mean_contrib = sum(recent_contrib) / len(recent_contrib)
            print(f"    Avg contribution ratio (last {len(recent_contrib)}): {mean_contrib:.3f}")

        print(f"  ═══════════════════════════════════════════\n")

        super().on_exit(agent)

    def step(self):
        """
        Called mỗi learning step.
        Decay entropy coefficient.
        """
        # Decay entropy
        self._current_entropy_coef = max(
            self._current_entropy_coef * self.hp['entropy_decay_rate'],
            self.hp['entropy_coef_end']
        )

        super().step()

    # ══════════════════════════════════════════════════════════
    # PHASE PARAMETERS
    # ══════════════════════════════════════════════════════════

    def get_parameters(self) -> PhaseParameters:
        """
        Proposal FROZEN, Refine ACTIVE
        """
        return PhaseParameters(
            alpha=self.hp['alpha'],
            beta=self.hp['beta'],
            entropy_coef_proposal=0.0,  # Không cần (frozen)
            entropy_coef_refine=self._current_entropy_coef,
            train_proposal=False,  # ← FROZEN
            train_refine=True,  # ← ACTIVE
            freeze_proposal=True,  # ← FROZEN
            freeze_refine=False,
            grad_clip_proposal=self.hp['grad_clip_proposal'],
            grad_clip_refine=self.hp['grad_clip_refine'],
            lr_proposal=self.hp['lr_proposal'],
            lr_refine=self.hp['lr_refine'],
        )

    # ══════════════════════════════════════════════════════════
    # LOSS COMPUTATION
    # ══════════════════════════════════════════════════════════

    def compute_losses(self, agent, batch_data) -> Dict[str, torch.Tensor]:
        """
        Chỉ tính loss cho Refine (Proposal frozen)

        Logic:
        - final = prop (frozen) + beta * delta (active)
        - Chỉ backprop vào delta
        - Entropy bonus cao để encourage exploration
        """
        # Unpack
        prop_logits = batch_data['prop_logits']
        delta_logits = batch_data['delta_logits']
        act_cat = batch_data['act_cat']
        advantages = batch_data['b_adv']
        old_log_probs = batch_data['b_old_lp']
        masks_exp = batch_data['masks_exp']
        batch_idx = batch_data['batch_idx']
        B_sub = batch_data['B_sub']
        # b_t_lens = batch_data['b_t_lens']  # ĐÃ BỎ: Không cần dùng nữa

        beta = self.hp['beta']

        # ══════════════════════════════════════════
        # REFINE LOSS ONLY (stop-gradient trên proposal)
        # ══════════════════════════════════════════

        # SỬA 1: Bỏ dấu gạch dưới (_) ở hàm mask_and_sanitize
        final_R = agent.mask_and_sanitize(
            prop_logits.detach() + beta * delta_logits,  # ← prop.detach() = frozen
            masks_exp
        )
        dist_R = Categorical(logits=final_R)

        # ══════════════════════════════════════════
        # PPO Loss
        # ══════════════════════════════════════════
        sum_lp_R = torch.zeros(B_sub, device=agent.device).scatter_add_(
            0, batch_idx, dist_R.log_prob(act_cat)
        )

        # SỬA 2: KHÔNG chia cho b_t_lens. Trừ thẳng luôn!
        ratio_R = torch.exp(sum_lp_R - old_log_probs)

        ppo_loss_R = -torch.min(
            ratio_R * advantages,
            torch.clamp(ratio_R, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages
        ).mean()

        # ══════════════════════════════════════════
        # Entropy bonus
        # ══════════════════════════════════════════
        sum_ent_R = torch.zeros(B_sub, device=agent.device).scatter_add_(
            0, batch_idx, dist_R.entropy()
        )

        # SỬA 3: KHÔNG chia cho b_t_lens. Dùng luôn sum_ent_R.mean()
        entropy_loss_R = -self._current_entropy_coef * sum_ent_R.mean()

        # Total loss cho Refine
        loss_refine = ppo_loss_R + entropy_loss_R

        # ══════════════════════════════════════════
        # TRACKING (cho on_exit summary)
        # ══════════════════════════════════════════
        with torch.no_grad():
            delta_norm = delta_logits.norm(dim=-1).mean().item()
            self._delta_norm_history.append(delta_norm)

            # Contribution ratio: |Δ| / (|prop| + |Δ|)
            prop_norm = prop_logits.norm(dim=-1).mean()
            delta_norm_t = delta_logits.norm(dim=-1).mean()
            contrib_ratio = (delta_norm_t / (prop_norm + delta_norm_t + 1e-8)).item()
            self._contribution_ratio_history.append(contrib_ratio)

        return {
            'loss_proposal': None,  # Không train proposal
            'loss_refine': loss_refine,
            'metrics': {
                'beta': beta,
                'entropy_R': sum_ent_R.mean().item(),  # Đổi thành sum_ent_R
                'entropy_coef': self._current_entropy_coef,
                'delta_norm': delta_norm,
                'contribution_ratio': contrib_ratio,
                'ppo_loss_R': ppo_loss_R.item(),
            }
        }

    # ══════════════════════════════════════════════════════════
    # TRANSITION LOGIC
    # ══════════════════════════════════════════════════════════

    def should_transition(self, metrics: Dict[str, float]) -> Optional[str]:
        """Phase 2 là phase cuối, không chuyển nữa"""
        # Track reward để log trong on_exit
        if 'avg_reward' in metrics:
            self._reward_history.append(metrics['avg_reward'])
        return None

    # ══════════════════════════════════════════════════════════
    # SERIALIZATION
    # ══════════════════════════════════════════════════════════

    def save_state(self) -> Dict[str, Any]:
        """Save state bao gồm tracking history và entropy"""
        state = super().save_state()
        state.update({
            'delta_norm_history': self._delta_norm_history.copy(),
            'contribution_ratio_history': self._contribution_ratio_history.copy(),
            'reward_history': self._reward_history.copy(),
            'current_entropy_coef': self._current_entropy_coef,
        })
        return state

    def load_state(self, state: Dict[str, Any]):
        """Load state bao gồm tracking history và entropy"""
        super().load_state(state)
        self._delta_norm_history = state.get('delta_norm_history', []).copy()
        self._contribution_ratio_history = state.get('contribution_ratio_history', []).copy()
        self._reward_history = state.get('reward_history', []).copy()
        self._current_entropy_coef = state.get('current_entropy_coef', self.hp['entropy_coef_start'])