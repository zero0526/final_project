from typing import TypedDict, Dict, Any
from src.configs.configs import cfg


class TimeManager:
    def __init__(self, neuron_cfg=cfg.hyper_neural):
        """
        Time managent for  HMDP model  (Hierarchical Markov Decision Process).
        """
        if neuron_cfg['TIME_SLOT_PER_TIMEFRAME'] <= 0:
            raise ValueError("slots_per_frame must be > 0")

        # Constants
        self.slot_duration: float = float(neuron_cfg['SLOT_DURATION'])
        self.slots_per_frame: int = int(neuron_cfg['TIME_SLOT_PER_TIMEFRAME'])
        self.max_steps: int = int(neuron_cfg['TIME_STEP_OFEP'])

        # State variables
        self.current_slot: int = 0
        self.current_frame: int = 0
        self.time_elapsed: float = 0.0

    def reset(self) -> None:
        """Đưa đồng hồ về trạng thái ban đầu (Slot 0, Frame 0)."""
        self.current_slot = 0
        self.current_frame = 0
        self.time_elapsed = 0.0

    def tick(self) -> None:
        """
        Chuyển sang Time Slot tiếp theo.
        Thường được gọi ở cuối hàm step() của Environment.
        """
        self.current_slot += 1

        # update Frame index
        self.current_frame = self.current_slot // self.slots_per_frame

        # Tính lại thời gian thực (Dùng phép nhân để tránh sai số cộng dồn float)
        self.time_elapsed = self.current_slot * self.slot_duration

    def is_new_frame(self) -> bool:
        """
        if new frame active Upper Agent (Service Placement).
        """
        if self.is_done():
            return False
        return (self.current_slot % self.slots_per_frame) == 0

    def is_done(self) -> bool:
        return self.current_slot >= self.max_steps

    def get_relative_slot(self) -> int:
        """
            order of the slot in the frame
        """
        return self.current_slot % self.slots_per_frame

    def to_abs_time(self, timeslot: int):
        return self.slot_duration * timeslot

    def get_state(self) -> Dict[str, Any]:
        """Trả về snapshot trạng thái dạng dict để log hoặc observe."""
        return {
            "global_slot": self.current_slot,
            "frame_idx": self.current_frame,
            "local_slot": self.get_relative_slot(),
            "time_elapsed_sec": round(self.time_elapsed, 4),
            "is_new_frame": self.is_new_frame()
        }

    def __repr__(self) -> str:
        status = "DONE" if self.is_done() else "RUNNING"
        rel_slot = self.get_relative_slot()
        return (f"<TimeManager [{status}] "
                f"Frame: {self.current_frame} | "
                f"Slot: {self.current_slot} (Local: {rel_slot}/{self.slots_per_frame}) | "
                f"Time: {self.time_elapsed:.2f}s>")

