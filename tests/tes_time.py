from src.envs.time_manager import TimeManager
import random

if __name__ == "__main__":
    time_manager = TimeManager()
    slot= random.randint(30, 30)
    for i in range(slot):
        time_manager.tick()
    print(slot)
    print(time_manager.get_state())