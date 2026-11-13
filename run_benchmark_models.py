import os
import argparse
import torch
import numpy as np
from config import get_config
from src.dataset import get_dataloaders
from src.models.baselines import TransRR, HybridCNNLSTM, ConvMixer, Resptrack1DCNN
from src.losses import CorrelationAwareLoss
from src.efficiency import compute_metrics
from src.vmd_utils import apply_vmd_batch

def get_model(name):
    name_clean = name.upper().replace("-", "_").replace(" ", "_")
    if "TRANSRR" in name_clean:
        return TransRR()
    elif "HYBRID" in name_clean or "LSTM" in name_clean:
        return HybridCNNLSTM()
    elif "CONVMIXER" in name_clean:
        return ConvMixer()
    elif "RESPTRACK" in name_clean or "1DCNN" in name_clean:
        return Resptrack1DCNN()
    else:
        raise ValueError(f"Model benchmark '{name}' tidak dikenali.")

def train_single_benchmark(model_name, cfg, train_loader, val_loader, device, alpha_pcc=2.0, use_vmd=False):
    print("\n" + "=" * 65)
    print(f" PELATIHAN BENCHMARK: {model_name} (VMD: {use_vmd})")
    print("=" * 65)

    model = get_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    criterion = CorrelationAwareLoss(alpha_pcc=alpha_pcc)

    best_score = float("inf")
    patience = cfg.train.patience
    patience_cnt = 0
    save_name = f"{model_name.lower().replace('-', '_')}_best.pth"
    save_path = os.path.join(cfg.CHECKPOINT_DIR, save_name)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_vmd:
                xb, _ = apply_vmd_batch(xb)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)

        scheduler.step()

        # Validasi
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                if use_vmd:
                    xb, _ = apply_vmd_batch(xb)
                out = model(xb)
                v_preds.extend(out.cpu().numpy().flatten())
                v_trues.extend(yb.numpy().flatten())

        m = compute_metrics(np.array(v_trues), np.array(v_preds))
        val_mae, val_pcc = m["MAE"], m["PCC"]
        val_score = val_mae + alpha_pcc * (1.0 - val_pcc)

        if val_score < best_score:
            best_score = val_score
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            stat = f"[BEST -> MAE: {val_mae:.3f} | PCC: {val_pcc:.3f} | Acc@2bpm: {m['Acc@2bpm(%)']:.1f}%]"
        else:
            patience_cnt += 1
            stat = f"[Patience: {patience_cnt}/{patience}] (PCC: {val_pcc:.3f})"

        if epoch % 5 == 0 or epoch == 1 or patience_cnt == 0:
            print(f"Epoch [{epoch:03d}/{cfg.train.epochs}] - Loss: {train_loss/len(train_loader.dataset):.4f} - Val MAE: {val_mae:.3f} {stat}")

        if patience_cnt >= patience:
            print(f"[INFO] Early stopping pada epoch {epoch}.")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--alpha_pcc", type=float, default=2.0)
    parser.add_argument("--use_vmd", action="store_true", default=False, help="Aktifkan VMD selama pelatihan")
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.train.device)
    use_vmd = args.use_vmd or getattr(cfg, "USE_VMD", False)
    train_loader, val_loader, _ = get_dataloaders(cfg, dataset_name=cfg.ACTIVE_DATASET)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    models_to_train = cfg.ACTIVE_BENCHMARK_MODELS if args.model.lower() == "all" else [args.model]

    for m_name in models_to_train:
        train_single_benchmark(m_name, cfg, train_loader, val_loader, device, alpha_pcc=args.alpha_pcc, use_vmd=use_vmd)

if __name__ == "__main__":
    main()