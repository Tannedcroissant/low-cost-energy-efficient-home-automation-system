"""
OFFLINE TRAINING PIPELINE (UK-DALE)
---------------------------------------------------
the script processes the raw UK-DALE .h5 dataset, synchronizes it 
with environmental parameters, applies Fixed Sliding Time Windows (FSTW), 
and trains the 1D-DCNN + BLSTM forecasting model on GPU infrastructure.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.forecasting_pipeline import HybridForecaster


WINDOW_SIZE = 288 
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_ukdale(h5_path):
    print(f"📥 Loading raw UK-DALE dataset from: {h5_path}")
    try:
        
        df = pd.read_hdf(h5_path, key='/building1/elec/meter1')
    except Exception as e:
        print(f"❌ Error reading .h5 file: {e}")
        print("Ensure 'ukdale.h5' is in the 'data/' directory.")
        exit(1)

    print("⚙️ Resampling raw high-frequency data to 5-minute intervals...")
    df_resampled = df.resample('5min').mean().fillna(method='ffill')
    
    
    power_data = df_resampled.iloc[:, 0].values.astype(np.float32)
    
    
    mean_val, std_val = power_data.mean(), power_data.std()
    normalized_data = (power_data - mean_val) / std_val
    
    return normalized_data

def apply_fstw(data, window_size):
    print("🪟 Applying Fixed Sliding Time Window (FSTW)...")
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def main():
    print(f"🚀 Initializing Offline Training on: {DEVICE}")
    
    
    h5_file = "./data/ukdale.h5"
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("⚠️ Please place the 'ukdale.h5' file inside the './data/' directory and run again.")
        return

    data = load_and_preprocess_ukdale(h5_file)
    X, y = apply_fstw(data, WINDOW_SIZE)
    
    print("Applying 70/15/15 Temporal Split for Train/Val/andTest...")
    train_split = int(len(X) * 0.70)
    val_split = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]

    X_train_tensor = torch.tensor(X_train).unsqueeze(-1).to(DEVICE)
    y_train_tensor = torch.tensor(y_train).unsqueeze(-1).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = HybridForecaster(seq_length=WINDOW_SIZE, features=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    print(f"🧠 Commencing BLSTM Training for {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Training Loss (MSE): {total_loss/len(train_loader):.4f}")
            
    
    os.makedirs("models", exist_ok=True)
    save_path = "models/hybrid_forecaster_raw.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete. Raw weights saved to: {save_path}")
    print("➡️ Next Step: Run 'quantize_model.py' to compress weights for edge deployment.")

if __name__ == "__main__":
    main()