import requests
import yaml
import os

class HomeAssistantBridge:
    def __init__(self, config_path="/Users/apple/smart_home_thesis/config.yaml"):
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        self.base_url = config['ha_url']
        
        
        self.headers = {
            "Authorization": f"Bearer {config['ha_token']}",
            "Content-Type": "application/json",
        }

    def test_connection(self):
        url = f"{self.base_url}/api/"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                print("✅ Successfully connected to Home Assistant API!")
                return True
            else:
                print(f"❌ Connection failed. Status Code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Network Error: Could not reach {self.base_url}. Error: {e}")
            return False

    def push_sensor_state(self, entity_id, state, unit):
        url = f"{self.base_url}/api/states/{entity_id}"
        payload = {
            "state": str(state),
            "attributes": {
                "unit_of_measurement": unit,
                "friendly_name": entity_id.split('.')[-1].replace('_', ' ').title()
            }
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.status_code in [200, 201]

    def toggle_virtual_ac(self, turn_on: bool):
        
        service = "turn_on" if turn_on else "turn_off"
        url = f"{self.base_url}/api/services/input_boolean/{service}"
        
        
        payload = {"entity_id": "input_boolean.virtual_ac_unit"} 
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            action = "ON" if turn_on else "OFF"
            print(f"✅ Commanded Virtual AC to turn {action}.")
        return response.status_code == 200


if __name__ == "__main__":
    
    bridge = HomeAssistantBridge(config_path="config.yaml")
    
    if bridge.test_connection():
        bridge.push_sensor_state("sensor.virtual_indoor_temperature", 24.5, "°C")
        print("✅ Pushed Virtual Temperature to HA.")
        
    
        bridge.toggle_virtual_ac(turn_on=True)