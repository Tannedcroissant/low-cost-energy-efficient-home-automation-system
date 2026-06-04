import time
import pandas as pd
import paho.mqtt.client as mqtt


MQTT_HOST = "192.168.64.2"  
MQTT_PORT = 1883
MQTT_USER = "agent_user"
MQTT_PASS = "123456"

class ThesisDataReplayer:
    def __init__(self, csv_path="/Users/apple/smart_home_thesis/thesis_master_dataset.csv"):
        
        self.indoor_temp = 24.0
        self.co2_level = 400.0 
        self.alpha = 0.05  
        self.solar_factor = 0.002 
        
        try:
            print(f"📊 Loading dataset from {csv_path}...")
            self.dataset = pd.read_csv(csv_path)
            print(f"✅ Successfully loaded {len(self.dataset)} rows of historical data.")
        except FileNotFoundError:
            print(f"❌ Error: Could not find {csv_path}. Please ensure the file exists.")
            exit(1)

        # connectung to mqtt Bbus
        self.mqtt = mqtt.Client()
        self.mqtt.username_pw_set(MQTT_USER, MQTT_PASS)
        try:
            
            clean_host = MQTT_HOST.strip()
            clean_port = int(MQTT_PORT)
            
            print(f" Attempting connection to: '{clean_host}' on port {clean_port}...")
            self.mqtt.connect(clean_host, clean_port, 60)
            
            self.mqtt.loop_start()
            print("✅ Connected to Home Assistant MQTT Bus.")
            self.mqtt.connect(MQTT_HOST, MQTT_PORT, 60)
            self.mqtt.loop_start()
            print("✅ Connected to Home Assistant MQTT Bus.")
        except Exception as e:
            print(f"❌ MQTT Connection Error: {e}")
            exit(1)

    def run_simulation(self, tick_rate_seconds=2):
    
        print("🌍 Starting Historical Data Replay...")
        
        for index, row in self.dataset.iterrows():
            print(f"\n--- 🕒 Historical Time: {row['timestamp']} ---")
            solar_heat = row['solar_irradiance'] * self.solar_factor
            heat_gain = self.alpha * (row['outdoor_temp'] - self.indoor_temp)
            self.indoor_temp = self.indoor_temp + heat_gain + solar_heat
            
            if row['occupancy'] > 0:
                self.co2_level = min(2000.0, self.co2_level + 50.0) # humans arw there
            else:
                self.co2_level = max(400.0, self.co2_level - 20.0) 
                
           
            self.mqtt.publish("smart_home/sensors/outdoor_temp", str(row['outdoor_temp']))
            self.mqtt.publish("smart_home/sensors/solar_irradiance", str(row['solar_irradiance']))
            self.mqtt.publish("smart_home/sensors/grid_price", str(row['grid_price']))
            self.mqtt.publish("smart_home/sensors/occupancy", str(row['occupancy']))
            self.mqtt.publish("smart_home/sensors/indoor_temp", f"{self.indoor_temp:.2f}")
            self.mqtt.publish("smart_home/sensors/co2_level", f"{self.co2_level:.1f}")
            print(f" solar: {row['solar_irradiance']} W/m² | Outdoor temp: {row['outdoor_temp']}°C -> Indoor: {self.indoor_temp:.2f}°C")
            print(f"🧍 Occupancy: {row['occupancy']} | 💨 CO2: {self.co2_level} ppm | 💰 Tariff: {row['grid_price']}x")
            
           
            time.sleep(tick_rate_seconds)

if __name__ == "__main__":
    replayer = ThesisDataReplayer()
    replayer.run_simulation(tick_rate_seconds=5)