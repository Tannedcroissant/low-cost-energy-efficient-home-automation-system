import os
import time
import requests
import datetime
import numpy as np
import torch
import paho.mqtt.client as mqtt
from collections import deque
from networks import QNetwork
from forecasting_pipeline import HybridForecaster 

MQTT_HOST = 'core-mosquitto'
MQTT_PORT = 1883
MQTT_USER = 'agent_user' 
MQTT_PASS = '123456'      
HA_URL = "http://supervisor/core/api"
HA_TOKEN = os.environ.get('SUPERVISOR_TOKEN')

class DualBrainEdgeAI:
    def __init__(self, dqn_path="models/dqn_smart_home.pth", blstm_path="models/hybrid_forecaster.pth"):
        print("🚀 [FINAL EVOLUTION] Booting Dual-Brain Edge AI...")
        self.device = torch.device("cpu")
        try:
            self.forecaster = HybridForecaster(seq_length=288, features=1).to(self.device)
            self.forecaster.load_state_dict(torch.load(blstm_path, map_location=self.device))
            self.forecaster.eval()
            print("🔮 1D-DCNN+BLSTM Forecaster loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load Forecaster: {e}")
            exit(1)
            
        # lodaing secon brain 
        try:
            self.dqn = QNetwork(state_size=8, action_size=8).to(self.device)
            self.dqn.load_state_dict(torch.load(dqn_path, map_location=self.device))
            self.dqn.eval()
            print("🧠 8D Deep Q-Network loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load DQN: {e}")
            exit(1)
        self.current_temp = 24.0
        self.co2_level = 400.0
        self.occupancy = 1.0
        self.solar = 0.0
        self.price = 0.12
        self.wm_state = 0 
        self.wm_timer = 0
        self.load_memory = deque([0.5] * 288, maxlen=288) 
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT Connected. Listening to Master Data Stream...")
            self.mqtt_client.subscribe("smart_home/sensors/#")

    def on_message(self, client, userdata, msg):
        try:
            payload = float(msg.payload.decode('utf-8'))
            topic = msg.topic
            if "indoor_temp" in topic: self.current_temp = payload
            elif "co2_level" in topic: self.co2_level = payload
            elif "occupancy" in topic: 
                self.occupancy = payload
                current_load = 0.5 + (self.occupancy * 1.5)
                self.load_memory.append(current_load)
            elif "solar" in topic: self.solar = payload
            elif "grid_price" in topic: self.price = payload
        except: pass

    def trigger_ha_action(self, entity_id, turn_on: bool):
        headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
        service = "turn_on" if turn_on else "turn_off"
        domain = entity_id.split('.')[0]
        try:
            requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json={"entity_id": entity_id})
        except: pass

    def run_inference_loop(self):
        self.mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
        self.mqtt_client.loop_start()
        print("⚡ Dual-Brain AI is ONLINE.")
        
        while True:
            memory_array = np.array(self.load_memory, dtype=np.float32).reshape(1, 288, 1)
            memory_tensor = torch.tensor(memory_array).to(self.device)
            
            with torch.no_grad():
                predicted_load = self.forecaster(memory_tensor).item()
            now = datetime.datetime.now()
            time_of_day = now.hour + (now.minute / 60.0)
            
            state = np.array([
                self.current_temp, time_of_day, self.price, self.wm_state,
                self.co2_level, self.occupancy, self.solar, predicted_load
            ], dtype=np.float32)
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.dqn(state_tensor)
            action = np.argmax(q_values.cpu().numpy())
            ac_on, start_wm, fan_on = action in [1,4,5,7], action in [2,4,6,7], action in [3,5,6,7]
            
            self.trigger_ha_action("input_boolean.virtual_ac", ac_on)
            self.trigger_ha_action("input_boolean.virtual_fan", fan_on)
            
            if start_wm and self.wm_state == 0:
                self.trigger_ha_action("input_boolean.virtual_washing_machine", True)
                self.wm_state, self.wm_timer = 1, 12
            if self.wm_state == 1:
                self.wm_timer -= 1
                if self.wm_timer <= 0:
                    self.wm_state = 2
                    self.trigger_ha_action("input_boolean.virtual_washing_machine", False)
            
            print(f" BLSTM 1-Hour forecast Load: {predicted_load:.2f} kW")
            print(f"temp: {self.current_temp:.1f}°C | 💨 CO2: {self.co2_level} | 💰 Price: {self.price}x")
            print(f"🤖 DQN Action -> Aircon: {'ON' if ac_on else 'OFF'} | Fan: {'ON' if fan_on else 'OFF'} | WM: {self.wm_state}\n")
            
            time.sleep(5)

if __name__ == "__main__":
    agent = DualBrainEdgeAI()
    agent.run_inference_loop()