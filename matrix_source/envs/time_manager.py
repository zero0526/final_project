class TimeManager:
    """
    NHIỆM VỤ CỦA THÀNH PHẦN (TIME MANAGER):
    1. Quản lý thời gian toàn cục (Absolute Time) và đếm bước (Step Counter).
    2. Đồng bộ hóa giữa Time Slot và Time Frame (Upper/Lower level).
    3. Cung cấp các flag trạng thái (New Frame, Done) dưới dạng scalar hoặc tensor nếu chạy song song nhiều môi trường.
    4. Tích hợp trực tiếp vào `MatrixSixGEnvironment` để điều phối việc tick thời gian.
    """
    def __init__(self, slot_duration, timeframe_size, max_steps):
        self.slot_duration = slot_duration
        self.timeframe_size = timeframe_size
        self.max_steps = max_steps
        self.current_step = 0
        self.time_elapsed = 0.0

    def tick(self):
        self.current_step += 1
        self.time_elapsed += self.slot_duration
        
    def is_new_frame(self):
        return self.current_step % self.timeframe_size == 0
