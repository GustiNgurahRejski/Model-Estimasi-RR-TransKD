import os
import time
import torch
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    errors = np.abs(y_true - y_pred)
    
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pcc = 0.0 if (np.std(y_true) == 0 or np.std(y_pred) == 0) else float(np.corrcoef(y_true, y_pred)[0, 1])
    acc_2bpm = float(np.mean(errors <= 2.0) * 100.0)
    
    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "Acc@2bpm(%)": acc_2bpm}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_model_size_mb(model, temp_path="temp_size_eval.pth"):
    torch.save(model.state_dict(), temp_path)
    size_bytes = os.path.getsize(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return {"size_kb": size_bytes / 1024, "size_mb": size_bytes / (1024 * 1024)}

def measure_inference_latency(model, input_shape=(1, 1, 750), device="cpu", simulate_wearable=True, num_runs=50):
    if simulate_wearable:
        torch.set_num_threads(1)
        target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)
        
    model = model.to(target_device)
    model.eval()
    dummy = torch.randn(input_shape).to(target_device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
            
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            
    return float(np.median(latencies))

def run_efficiency_report(model, input_shape=(1, 1, 750), device="cpu", simulate_wearable=True):
    params = count_parameters(model)
    size_dict = compute_model_size_mb(model)
    latency = measure_inference_latency(model, input_shape, device, simulate_wearable)
    
    print("\n--- Ringkasan Efisiensi Hardware Wearable ---")
    print(f"Total Parameters     : {params:,}")
    print(f"Model Size           : {size_dict['size_kb']:.2f} KB ({size_dict['size_mb']:.4f} MB)")
    print(f"Inference Latency    : {latency:.2f} ms/sample (Wearable Mode: {simulate_wearable})")
    print("---------------------------------------------\n")
    return {"parameters": params, "size_kb": size_dict["size_kb"], "size_mb": size_dict["size_mb"], "latency_ms": latency}