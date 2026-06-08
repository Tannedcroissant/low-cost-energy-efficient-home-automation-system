from src.rl_agent.environment import SmartHomeEnv

def run_random_agent():
    env = SmartHomeEnv()
    obs, info = env.reset()
    
    total_reward = 0
    print(f"Starting Episode. Initial Temp: {obs[0]:.2f}°C")
    
    for _ in range(env.max_steps):
        random_action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        
    print(f"End of 24 Hours.")
    print(f"Final Temp: {obs[0]:.2f}°C")
    print(f"Total Accumulated Reward: {total_reward:.2f}")
    
if __name__ == "__main__":
    run_random_agent()