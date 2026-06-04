import torch
import numpy as np
from environment import SmartHomeEnv
from networks import QNetwork

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def evaluate_agent(model_path="../../models/dqn_smart_home.pth"):
    env = SmartHomeEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    q_network = QNetwork(state_size, action_size).to(DEVICE)
    try:
        q_network.load_state_dict(torch.load(model_path, map_location=DEVICE))
        q_network.eval() 
        print("🧠 Successfully loaded the Multi-Objective Deep Q-Network.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find model at {model_path}")
        return
        
    print("📊 Running 24-Hour Evaluation (Strict Exploitation)...\n")
    
    state, _ = env.reset()
    env.current_temp = 24.0 
    state[0] = 24.0
    
    total_reward = 0
    temperatures = []
    ac_on_count = 0
    total_energy_cost = 0
    wm_start_hour = None
    
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            q_values = q_network(state_tensor)
        action = np.argmax(q_values.cpu().numpy())
        
        if action in [2, 3] and env.wm_state == 0:
            wm_start_hour = (env.current_step * 5) / 60.0
            
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        temperatures.append(env.current_temp)
        if info["ac_state"]:
            ac_on_count += 1
        total_energy_cost += info["cost"]
            
        state = next_state
        total_reward += reward

    avg_temp = np.mean(temperatures)
    max_temp = np.max(temperatures)
    min_temp = np.min(temperatures)
    
    print(f"--- 📈 Final Multi-Objective Results ---")
    print(f"Total Accumulated Reward:   {total_reward:.2f}")
    print(f"Average Indoor Temperature: {avg_temp:.2f}°C")
    print(f"Temperature Range:          {min_temp:.2f}°C to {max_temp:.2f}°C")
    print(f"Total AC Operational Ticks: {ac_on_count} / {env.max_steps}")
    print(f"Total Energy Cost (Tokens): {total_energy_cost:.2f}")
    
    if wm_start_hour is not None:
        # Format the time nicely
        hours = int(wm_start_hour)
        mins = int((wm_start_hour - hours) * 60)
        time_str = f"{hours:02d}:{mins:02d}"
        
        print(f"Washing Machine Started:    {time_str}")
        if 14.0 <= wm_start_hour < 20.0:
            print("⚠️ WARNING: Agent ran laundry during PEAK hours (Failed to optimize time).")
        else:
            print("✅ SUCCESS: Agent successfully delayed laundry to OFF-PEAK hours!")
    else:
        print("❌ CATASTROPHIC FAILURE: Agent forgot to run the washing machine.")

if __name__ == "__main__":
    evaluate_agent()