import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# --- Path Resolution ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AIRFOIL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

sys.path.append(os.path.join(AIRFOIL_DIR, 'models'))
from MatrixKan import MatrixKAN

# Access the local fixed_point_utils
sys.path.append(os.path.join(CURRENT_DIR, 'fixed_point_utils'))
import hdl_emulator_bspline
import quantization_utils

# --- Configuration ---
CHECKPOINT_PATH = os.path.join(AIRFOIL_DIR, "experiments", "saved_fp32_models", "bspline_fp32.pt")
DATA_PATH = os.path.join(AIRFOIL_DIR, "data", "nasa_airfoil_data.csv")

# Hardware testcase output directory
QUANT_RESULTS_DIR = os.path.join(AIRFOIL_DIR, "experiments", "quantization_results", "bspline_testcase")
os.makedirs(QUANT_RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(QUANT_RESULTS_DIR, "mem_init"), exist_ok=True)

WIDTH = [5, 7, 1] 
GRID = 4
K = 2

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
    print(f"Initializing MatrixKAN {WIDTH} and loading FP32 checkpoint...")
    model = MatrixKAN(
        width=WIDTH, 
        grid=GRID, 
        k=K, 
        device='cpu', 
        base_fun='zero', 
        symbolic_enabled=False, 
        auto_save=False
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True))
    model = model.float()
    model.eval()
    
    # --- WEIGHT EXTRACTION & HARDWARE SANITY CHECKS ---
    model_weights = [] 
    print("\nExtracting and Folding Weights...")
    for i, layer in enumerate(model.act_fun):
        coef = layer.coef.detach().cpu().numpy()
        scale_sp = layer.scale_sp.detach().cpu().numpy()
        folded_weights = coef * scale_sp[:, :, np.newaxis]
        model_weights.append(folded_weights)
        
        col0 = layer.grid_range[:, 0]
        col1 = layer.grid_range[:, 1]
        assert torch.all(col0 == -1.0), f"Layer {i} grid range min is not -1.0"
        assert torch.all(col1 == 1.0), f"Layer {i} grid range max is not 1.0"
        assert (layer.mask.detach() != 1).sum().item() == 0, f"Layer {i} contains sparse masking (not supported)"

    # --- FP32 BASELINE CHECK ---
    with torch.no_grad():
        X_calib = torch.from_numpy(X_val_scaled).float()
        torch_out_scaled = model(X_calib).numpy().flatten()
        fp32_preds = (torch_out_scaled * y_std) + y_mean
    
    fp32_r2 = r2_score(y_v, fp32_preds)
    fp32_acc = max(0.0, fp32_r2 * 100)

    # 3. Hardware Emulation Loop (INT16)
    print(f"\nStarting Fixed-Point Inference (INT16) on {len(X_val_scaled)} samples...")
    int16_preds = []

    for i in range(len(X_val_scaled)):
        sample_np = X_val_scaled[i:i+1].astype(np.float32) 
        
        # Trigger HDL dump on the very first sample only
        save_dir = QUANT_RESULTS_DIR if i == 0 else None
        
        q_out = hdl_emulator_bspline.forward_model(sample_np, model_weights, save_dir=save_dir)
        int16_preds.append((q_out.item() * y_std) + y_mean)

        if (i+1) % 100 == 0:
            print(f" Processed {i+1}/{len(X_val_scaled)} samples...")

    int16_preds = np.array(int16_preds)
    int16_r2 = r2_score(y_v, int16_preds)
    int16_acc = max(0.0, int16_r2 * 100)
    
    print("\n" + "="*40)
    print("   AIRFOIL ISO-ACCURACY BENCHMARK")
    print("="*40)
    print(f"Architecture:      MatrixKAN {WIDTH}")
    print(f"FP32 Accuracy:     {fp32_acc:.2f}%")
    print(f"INT16 Accuracy:    {int16_acc:.2f}%")
    print("-" * 40)
    print(f"ACCURACY DROPOFF:  {fp32_acc - int16_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_comparison_analysis()