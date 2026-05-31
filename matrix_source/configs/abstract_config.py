from abc import ABC
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union


# ======================== CONFIG CLASSES ========================
@dataclass
class BasePhaseConfig:
    k_epochs: int = 3
    clip_eps: float = 0.3
    tau: float = 0.005
    temperature: Tuple[float, float] = (1.0, 1.0)
    lr_actor: Union[float, Tuple[float, float]] = 1e-4
    lr_critic: Union[float, Tuple[float, float]] = 5e-4


@dataclass
class EntropyConfig:
    agent_type: str = 'lower'
    explore_start: float = 0.05
    explore_end: float = 0.03
    refine_end: float = 0.01
    focus_end: float = 0.005
    freeze_end: float = 0.003
    rewarm_boost: float = 0.01
    min_coef: float = 0.003
    rewarm_until_phase: Optional[str] = 'EXPLORE'


# ======================== ABSTRACT BASE CONFIG ========================
class AbstractTrainingConfig(ABC):
    PHASES: list = ['EXPLORE', 'REFINE', 'CONVERGE']

    # Sử dụng Class Variables thay cho @property
    PHASE_BOUNDS: Dict[str, Tuple[int, int]]
    MAX_CYCLES: int
    CONVERGE_START: int

    UPPER_PHASES: Dict[str, BasePhaseConfig]
    LOWER_PHASES: Dict[str, BasePhaseConfig]

    UPPER_ENTROPY: EntropyConfig
    LOWER_ENTROPY: EntropyConfig

    UPPER_BUFFER: Dict = {'min_size': 512, 'batch_size': 64}
    LOWER_BUFFER: Dict = {'min_size': 4096, 'batch_size': 128}

    # ─── Helpers ───
    @classmethod
    def get_phase(cls, cycle: int) -> str:
        bounds = cls.PHASE_BOUNDS
        if cycle <= bounds['EXPLORE'][1]:
            return 'EXPLORE'
        elif cycle <= bounds['REFINE'][1]:
            return 'REFINE'
        return 'CONVERGE'

    @classmethod
    def get_phase_progress(cls, cycle: int, phase: str) -> float:
        start, end = cls.PHASE_BOUNDS[phase]
        return (cycle - start) / max(1, end - start)

    @classmethod
    def is_converge(cls, cycle: int) -> bool:
        return cycle >= cls.CONVERGE_START