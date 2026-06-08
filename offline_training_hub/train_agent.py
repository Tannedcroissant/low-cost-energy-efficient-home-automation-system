import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.rl_agent.environment import SmartHomeEnv
from src.rl_agent.dqn_agent import DQNAgent

def train_agent(episodes=500):
    env = SmartHomeEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    epsilons = []
    
    print(f" Starting RL Training for {episodes} Simulated Days...")
    print("This will only take a minute or two!\n")

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn(batch_size=64)
            
            state = next_state
            total_reward += reward
        
        if e % 10 == 0:
            agent.update_target_network()
        agent.decay_epsilon()
    
        scores.append(total_reward)
        epsilons.append(agent.epsilon)
        
        if (e + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"🗓️ Day {e + 1}/{episodes} | Avg Reward (Last 50): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("\nTraining Complete!")
    return agent, scores

def save_artifacts(agent, scores):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    torch.save(agent.q_network.state_dict(), "models/dqn_smart_home.pth")
    print(" Model saved to models/dqn_smart_home.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(scores, color='blue', alpha=0.6)
    
    moving_avg = np.convolve(scores, np.ones(50)/50, mode='valid')
    plt.plot(np.arange(49, len(scores)), moving_avg, color='red', linewidth=2, label='50-Day Moving Average')
    
    plt.title('Deep Q-Network Learning Curve: Smart Home Energy Management', fontsize=14)
    plt.xlabel('Simulated Days (Episodes)', fontsize=12)
    plt.ylabel('Accumulated Reward (Energy & Comfort)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("results/figures/learning_curve.png", dpi=300, bbox_inches='tight')
    print("📊 Graph saved to results/figures/learning_curve.png")

if __name__ == "__main__":
    trained_agent, reward_history = train_agent(episodes=500)

    save_artifacts(trained_agent, reward_history)