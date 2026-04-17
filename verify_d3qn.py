import torch
import numpy as np
from src.agents.d3qn import D3QNAgent

def test_exclusion():
    state_dim = 10
    action_dim = 5
    u_action_dim = 32
    agent = D3QNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        u_action_dim=u_action_dim,
        mf_hidden_sizes=(16,),
        mf_lr=1e-3,
        exclude_zero=True
    )
    
    state = np.random.rand(state_dim)
    prev_mf = np.random.rand(action_dim)
    
    print("Testing exploration (epsilon=1.0)...")
    counts = {i: 0 for i in range(u_action_dim)}
    for _ in range(1000):
        action = agent.choose_action(state, prev_mf, epsilon=1.0, zeta=1.0)
        counts[action] += 1
    
    print(f"Action 0 count: {counts[0]}")
    if counts[0] == 0:
        print("SUCCESS: Action 0 never selected in exploration.")
    else:
        print("FAILURE: Action 0 selected in exploration!")

    print("\nTesting exploitation (epsilon=0.0)...")
    # Force Q-values to be high for index 0 if not penalized
    with torch.no_grad():
        # This is tricky because Q-values are from the network.
        # But we can check the logic by calling with low epsilon many times.
        found_zero = False
        for _ in range(100):
            action = agent.choose_action(state, prev_mf, epsilon=0.0, zeta=1.0)
            if action == 0:
                found_zero = True
                break
        
        if not found_zero:
            print("SUCCESS: Action 0 never selected in exploitation.")
        else:
            print("FAILURE: Action 0 selected in exploitation!")

if __name__ == "__main__":
    test_exclusion()
