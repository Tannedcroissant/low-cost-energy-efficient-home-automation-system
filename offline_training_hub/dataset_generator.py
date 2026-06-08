"""
This script generates a lightweight, SYNTHETIC dummy dataset. 
It is provided solely to allow reviewers to test the execution of the 
machine learning pipeline (1D-DCNN, BLSTM, DQN) and Home Assistant 
integration without needing to download the huge UK-DALE .h5 dataset.

The actual model weights (.pth) provided in this repository were trained 
on the genuine UK-DALE corpus offline.
"""
import pandas as pd
import numpy as np

def generate_synthetic_dataset(filename="thesis_master_dataset.csv", days=7, freq="5min"):
    print(f"🧬 Generating {days}-day synthetic smart home dataset...")
    start_date = "2023-07-01"
    time_index = pd.date_range(start=start_date, periods=(days * 24 * 60) // 5, freq=freq)
    
    df = pd.DataFrame(index=time_index)
    df['timestamp'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
    hours = df.index.hour + df.index.minute / 60.0
    base_temp = 27.0
    temp_amplitude = 7.0
    temp_noise = np.random.normal(0, 0.5, len(df))
    df['outdoor_temp'] = base_temp + temp_amplitude * np.sin((hours - 9) * np.pi / 12) + temp_noise
    df['outdoor_temp'] = df['outdoor_temp'].round(2)
    
    solar_wave = 850.0 * np.sin((hours - 6) * np.pi / 12)
    solar_noise = np.random.normal(0, 20, len(df))
    df['solar_irradiance'] = np.clip(solar_wave + solar_noise, 0, None)
    df.loc[(hours < 6) | (hours > 18), 'solar_irradiance'] = 0.0
    df['solar_irradiance'] = df['solar_irradiance'].round(0)
    df['occupancy'] = 1
    work_hours_mask = (hours >= 8.0) & (hours < 17.0)
    df.loc[work_hours_mask, 'occupancy'] = 0
    
    random_flips = np.random.rand(len(df)) < 0.05
    df.loc[random_flips, 'occupancy'] = 1 - df.loc[random_flips, 'occupancy']
    df['grid_price'] = 0.12
    peak_mask = (hours >= 14.0) & (hours < 20.0)
    df.loc[peak_mask, 'grid_price'] = 0.45
    
    df = df[['timestamp', 'outdoor_temp', 'solar_irradiance', 'occupancy', 'grid_price']]
    df.to_csv(filename, index=False)
    
    print(f"✅ Success! Created {filename} with {len(df)} rows of data.")
    print("\n--- Data Sample ---")
    print(df.head(10))

if __name__ == "__main__":
    generate_synthetic_dataset()