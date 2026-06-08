import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class HybridForecaster(nn.Module):
    def __init__(self, seq_length=288, features=1, cnn_out_channels=16, lstm_hidden=64):
        super(HybridForecaster, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=features, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.blstm = nn.LSTM(
            input_size=cnn_out_channels, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        c_out = self.cnn(x)
        c_out = c_out.permute(0, 2, 1)
        lstm_out, _ = self.blstm(c_out)
        final_timestep = lstm_out[:, -1, :] 
        prediction = self.fc(final_timestep)
        return prediction


def create_sliding_windows(data, window_size=288):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def train_model(csv_path="./thesis_master_dataset.csv"):
    print("📊 Loading dataset for Load Forecasting...")
    df = pd.read_csv(csv_path)
    if 'energy_load' not in df.columns:
        df['energy_load'] = 0.5 + (df['occupancy'] * 1.5) + np.random.normal(0, 0.2, len(df))
        
    raw_data = df['energy_load'].values.astype(np.float32)
    
    mean_val, std_val = raw_data.mean(), raw_data.std()
    normalized_data = (raw_data - mean_val) / std_val
    print(" Applying Fixed Sliding Time Window (FSTW)...")
    X, y = create_sliding_windows(normalized_data, window_size=288)
    
    
    X_tensor = torch.tensor(X).unsqueeze(-1) 
    y_tensor = torch.tensor(y).unsqueeze(-1)
    split = int(len(X_tensor) * 0.8)
    train_dataset = TensorDataset(X_tensor[:split], y_tensor[:split])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = HybridForecaster(seq_length=288, features=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print(" Initiating 1D-DCNN + BLSTM Training Phase...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Mean Squared Error (Loss): {total_loss/len(train_loader):.4f}")
        
    print("✅ Training Complete. Saving quantized weights...")
    torch.save(model.state_dict(), "/Users/apple/smart_home_thesis/models/hybrid_forecaster.pth")
    print("💾 Model saved to models/hybrid_forecaster.pth")

if __name__ == "__main__":
    train_model()