import os
import sys

# Ensure the models directory is in sys.path to support the new file structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Global imports for the custom layers from the models folder
from ChebyKANLayer import ChebyKANLayer
from fftKAN import NaiveFourierKANLayer
from MatrixKan import MatrixKAN

# ---------------------------------------------------------
# 1. Configuration & Model Selection
# ---------------------------------------------------------
MODEL_TYPE = 'bspline' # Options: 'chebyshev', 'fourier', 'bspline', 'mlp'
TARGET_ACCURACY = '93.5' 

EPOCHS = 200
LR = 0.01

# Update save directory to experiments/saved_fp32_models
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'saved_fp32_models'))
os.makedirs(SAVE_DIR, exist_ok=True)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# ---------------------------------------------------------
# 2. EXACT Architectures (Hardware Aligned)
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class Chebyshev(nn.Module):
    def __init__(self, layer_sizes, degree=3):
        super(Chebyshev, self).__init__()
        self.layers = nn.ModuleList([
            ChebyKANLayer(layer_sizes[i], layer_sizes[i+1], degree) 
            for i in range(len(layer_sizes) - 1)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Fourier(nn.Module):
    def __init__(self, layer_sizes, grid_size, layer_class):
        super(Fourier, self).__init__()
        self.layers = nn.ModuleList([
            layer_class(layer_sizes[i], layer_sizes[i+1], grid_size)
            for i in range(len(layer_sizes) - 1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(layer_sizes[i+1], elementwise_affine=False) 
            for i in range(len(layer_sizes) - 2)
        ])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(self.norms[i](x))
        return x

def get_optimal_model(model_type, target_acc):
    if model_type == 'bspline':
        if target_acc == '93.5':
            print("Initializing B-Spline KAN: [5, 7, 1] (Target: ~93.5%)")
            return MatrixKAN(width=[5, 7, 1], grid=4, k=2, device=device_str, symbolic_enabled=False, auto_save=False, base_fun='zero')
            
    elif model_type == 'chebyshev':
        if target_acc == '93.5':
            print("Initializing Chebyshev KAN: [5, 16, 1], Degree=3 (Target: ~93.5%)")
            return Chebyshev([5, 16, 1], degree=3)
            
    elif model_type == 'fourier':
        if target_acc == '93.5':
            print("Initializing Fourier KAN: [5, 12, 1], Grid Size=3 (Target: ~93.5%)")
            return Fourier([5, 12, 1], grid_size=3, layer_class=NaiveFourierKANLayer)
            
    elif model_type == 'mlp':
        if target_acc == '93.5':
            print("Initializing MLP: [5, 234, 78, 18, 1] (Target: ~93.5%)")
            return MLP([5, 234, 78, 18, 1])
    
    raise ValueError(f"Model config not found for TYPE: {model_type} and TARGET: {target_acc}")

# ---------------------------------------------------------
# 3. Data Loading 
# ---------------------------------------------------------
class NASADataset:
    def __init__(self, csv_file, device):
        self.data = pd.read_csv(csv_file)
        X_all = self.data.iloc[:, :5].values
        y_all = self.data.iloc[:, 5].values
        
        X_tr, X_v, y_tr, y_v = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        
        self.X_train = torch.tensor(X_tr, dtype=torch.float32).to(device)
        self.X_val = torch.tensor(X_v, dtype=torch.float32).to(device)
        
        # Scale input features to [-1, 1] matching Optuna preprocessing
        self.x_min = self.X_train.min(dim=0, keepdim=True)[0]
        self.x_max = self.X_train.max(dim=0, keepdim=True)[0]
        self.X_train = 2.0 * (self.X_train - self.x_min) / (self.x_max - self.x_min) - 1.0
        self.X_val = 2.0 * (self.X_val - self.x_min) / (self.x_max - self.x_min) - 1.0
        
        self.y_mean = y_tr.mean()
        self.y_std = y_tr.std()
        
        y_train_scaled = (y_tr - self.y_mean) / self.y_std
        y_val_scaled = (y_v - self.y_mean) / self.y_std
        
        self.y_train = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
        self.y_val = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1).to(device)
        self.y_val_orig = y_v 

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
def train_model():
    model = get_optimal_model(MODEL_TYPE, TARGET_ACCURACY)
    
    # Convert parameters to float32 to prevent Float vs Double errors (especially for MatrixKAN)
    model = model.float().to(device)
    
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'nasa_airfoil_data.csv'))
    print(f"Loading NASA dataset from {DATA_PATH}...")
    dataset = NASADataset(DATA_PATH, device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = -float('inf')
    save_path = ""
    
    print(f"--- Starting FP32 Training for {MODEL_TYPE.upper()} ({TARGET_ACCURACY} Target) ---")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        preds_train = model(dataset.X_train)
        loss = criterion(preds_train, dataset.y_train)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds_val_scaled = model(dataset.X_val)
            preds_val_real = (preds_val_scaled.cpu().numpy().flatten() * dataset.y_std) + dataset.y_mean
            
            r2 = r2_score(dataset.y_val_orig, preds_val_real)
            val_acc = max(0.0, r2 * 100)
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_fp32.pt")
            torch.save(model.state_dict(), save_path)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train MSE: {loss.item():.4f} | Val Accuracy: {val_acc:.2f}%")
            
    print(f"Training complete. Best model (Accuracy: {best_acc:.2f}%) saved to {save_path}")

if __name__ == "__main__":
    train_model()