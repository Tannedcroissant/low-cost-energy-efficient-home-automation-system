import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class SmartHomeEnv(gym.Env):
    def __init__(self, dataset_path="/Users/apple/smart_home_thesis/thesis_master_dataset.csv"):
        super(SmartHomeEnv, self).__init__()
        
        
        try:
            self.dataset = pd.read_csv(dataset_path)
            self.max_steps = 288 # 24 hours at 5-minute intervals
        except Exception as e:
            print(f"FATAL: Could not load {dataset_path}. run the data first.")
            exit(1)
            
       
        self.target_temp = 22.0
        self.alpha = 0.05  # Insulation thermal leak
        self.beta = 0.15   # AC cooling power
        self.solar_factor = 0.002 # Solar window gain
        
        self.ac_power = 1.5  # kW
        self.wm_power = 2.0  # kW
        self.fan_power = 0.2 # kW the fan
        
        self.wm_duration_steps = 12 # 60 mins
        
    
        self.action_space = spaces.Discrete(8)
        

        low = np.array([10.0, 0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([40.0, 24.0, 5.0, 2.0, 3000.0, 1.0, 1200.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        
        max_start = len(self.dataset) - self.max_steps - 1
        self.start_idx = np.random.randint(0, max(1, max_start))
        self.current_step = 0
        self.current_temp = 24.0
        self.co2_level = 400.0
        self.wm_state = 0
        self.wm_timer = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.dataset.iloc[self.start_idx + self.current_step]
        time_str = row['timestamp'].split(' ')[1]
        hours, mins, _ = map(int, time_str.split(':'))
        time_of_day = hours + (mins / 60.0)
        future_idx = min(self.start_idx + self.current_step + 12, len(self.dataset)-1)
        future_row = self.dataset.iloc[future_idx]
        predicted_load = 0.5 + (future_row['occupancy'] * 1.5) 

        return np.array([
            self.current_temp, 
            time_of_day, 
            row['grid_price'], 
            self.wm_state,
            self.co2_level,
            row['occupancy'],
            row['solar_irradiance'],
            predicted_load 
        ], dtype=np.float32)

    def step(self, action):
        row = self.dataset.iloc[self.start_idx + self.current_step]
        ac_on = action in [1, 4, 5, 7]
        start_wm = action in [2, 4, 6, 7]
        fan_on = action in [3, 5, 6, 7]
        if start_wm and self.wm_state == 0:
            self.wm_state = 1
            self.wm_timer = self.wm_duration_steps
            
        wm_power_used = 0.0
        if self.wm_state == 1:
            wm_power_used = self.wm_power
            self.wm_timer -= 1
            if self.wm_timer <= 0:
                self.wm_state = 2
        heat_gain = self.alpha * (row['outdoor_temp'] - self.current_temp)
        solar_heat = row['solar_irradiance'] * self.solar_factor
        cooling_effect = self.beta if ac_on else 0.0
        self.current_temp = self.current_temp + heat_gain + solar_heat - cooling_effect
        
        if row['occupancy'] > 0:
            self.co2_level += 50.0 # humans exhaling
        else:
            self.co2_level -= 10.0 # ifthere's a leak
            
        if fan_on:
            self.co2_level -= 100.0 
            
        self.co2_level = np.clip(self.co2_level, 400.0, 3000.0)
        total_power = (self.ac_power if ac_on else 0.0) + wm_power_used + (self.fan_power if fan_on else 0.0)
        energy_cost = total_power * row['grid_price']
        if row['occupancy'] > 0:
            discomfort = 2.0 * ((self.current_temp - self.target_temp) ** 2)
        else:
            discomfort = 0.5 * ((self.current_temp - self.target_temp) ** 2) 
        co2_penalty = 0.0
        if self.co2_level > 800.0:
            co2_penalty = (self.co2_level - 800.0) * 0.01 
            
        reward = -(energy_cost + discomfort + co2_penalty)
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        if terminated and self.wm_state != 2:
            reward -= 500.0 
            
        info = {"temp": self.current_temp, "co2": self.co2_level, "cost": energy_cost}
        return self._get_obs(), reward, terminated, False, info