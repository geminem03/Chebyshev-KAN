import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# --- Path Resolution ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AIRFOIL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

sys.path.append(os.path.join(AIRFOIL_DIR, 'models'))
from ChebyKANLayer import ChebyKANLayer

# Access the local fixed_point_utils
sys.path.append(os.path.join(CURRENT_DIR, 'fixed_point_utils'))
import hdl_emulator_chebyshev

# --- Configuration ---
CHECKPOINT_PATH = os.path.join(AIRFOIL_DIR, "experiments", "saved_fp32_models", "chebyshev_fp32.pt")
DATA_PATH = os.path.join(AIRFOIL_DIR, "data", "nasa_airfoil_data.csv")

# Hardware testcase output directory
QUANT_RESULTS_DIR = os.path.join(AIRFOIL_DIR, "experiments", "quantization_results", "chebyshev_testcase")
os.makedirs(QUANT_RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(QUANT_RESULTS_DIR, "mem_init"), exist_ok=True)

LAYER_SIZES = [5, 16, 1]
DEGREE = 3

# --- PyTorch Model Definition ---
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

def run_comparison_analysis():
    # 1. Load Data
    print(f"Loading NASA dataset from {DATA_PATH}...")
    data = pd.read_csv(DATA_PATH)
    X_all = data.iloc[:, :5].values
    y_all = data.iloc[:, 5].values
    
    # Matching train_fp32.py split
    X_tr, X_v, y_tr, y_v = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    X_train_min, X_train_max = X_tr.min(axis=0), X_tr.max(axis=0)
    X_val_scaled = 2.0 * (X_v - X_train_min) / (X_train_max - X_train_min) - 1.0
    
    y_mean, y_std = y_tr.mean(), y_tr.std()

    # 2. Load Torch Model
    print(f"Initializing Chebyshev KAN {LAYER_SIZES} and loading FP32 checkpoint...")
    model = Chebyshev(LAYER_SIZES, degree=DEGREE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True))
    model = model.float()
    model.eval()
    
    # 3. FP32 Baseline (Replaces Calibration)
    print("\nRunning FP32 Baseline on Validation Set...")
    with torch.no_grad():
        X_calib = torch.from_numpy(X_val_scaled).float()
        torch_out_scaled = model(X_calib).numpy().flatten()
        fp32_preds = (torch_out_scaled * y_std) + y_mean
    
    fp32_r2 = r2_score(y_v, fp32_preds)
    fp32_acc = max(0.0, fp32_r2 * 100)

    # 4. Extract Weights
    print("Extracting Chebyshev Coefficients...")
    chebykan1_weights = model.layers[0].cheby_coeffs.detach().cpu().numpy()
    chebykan2_weights = model.layers[1].cheby_coeffs.detach().cpu().numpy()
    cheby_weights = [chebykan1_weights, chebykan2_weights]

    # 5. Hardware Emulation Loop (INT16)
    print(f"\nStarting Fixed-Point Inference (INT16) on {len(X_val_scaled)} samples...")
    int16_preds = []

    for i in range(len(X_val_scaled)):
        sample_np = X_val_scaled[i:i+1].astype(np.float32) 
        
        # Trigger HDL dump on the very first sample only
        save_dir = QUANT_RESULTS_DIR if i == 0 else None
        
        _, q_out_dq = hdl_emulator_chebyshev.forward_airfoil(
            sample_np, 
            cheby_weights, 
            out_q16_frac=10,
            save_dir=save_dir
        )
        
        # Re-scale back to target distribution
        int16_preds.append((q_out_dq.item() * y_std) + y_mean)

        if (i+1) % 100 == 0:
            print(f" Processed {i+1}/{len(X_val_scaled)} samples...")

    int16_preds = np.array(int16_preds)
    int16_r2 = r2_score(y_v, int16_preds)
    int16_acc = max(0.0, int16_r2 * 100)
    
    print("\n" + "="*40)
    print("   AIRFOIL ISO-ACCURACY BENCHMARK")
    print("="*40)
    print(f"Architecture:      Chebyshev {LAYER_SIZES}, Degree={DEGREE}")
    print(f"FP32 Accuracy:     {fp32_acc:.2f}%")
    print(f"INT16 Accuracy:    {int16_acc:.2f}%")
    print("-" * 40)
    print(f"ACCURACY DROPOFF:  {fp32_acc - int16_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_comparison_analysis()