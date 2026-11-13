import os
import argparse
import torch
import numpy as np
from config import get_config
from src.dataset import get_dataloaders
from src.models.transformer import build_transformer
from src.losses import CorrelationAwareLoss
from src.efficiency import compute_metrics, run_efficiency_report
from src.vmd_utils import apply_vmd_batch

def main():
    parser = argparse.ArgumentParser(description="Pelatihan Model Teacher")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--alpha_pcc", type=float, default=2.0)
    parser.add_argument("--use_vmd", action="store_true", default=False, help="Aktifkan VMD pada input sinyal")
    args = parser.parse_args()

    cfg = get_config()
    epochs = args.epochs if args.epochs is not None else cfg.train.epochs
    device = torch.device(cfg.train.device)
    use_vmd = args.use_vmd or getattr(cfg, "USE_VMD", False)

    print("=" * 65)
    print(f" PELATIHAN TEACHER TRANSFORMER (VMD: {use_vmd})")
    print(f" Device: {device} | Dataset: {cfg.ACTIVE_DATASET}")
    print("=" * 65)

    torch.manual_seed(cfg.split.random_seed)
    np.random.seed(cfg.split.random_seed)

    train_loader, val_loader, _ = get_dataloaders(cfg, dataset_name=cfg.ACTIVE_DATASET)
    model = build_transformer(cfg.teacher_model, is_student=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CorrelationAwareLoss(alpha_pcc=args.alpha_pcc)

    best_score = float("inf")
    patience = cfg.train.patience
    patience_cnt = 0
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(cfg.CHECKPOINT_DIR, "teacher_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_vmd:
                xb, _ = apply_vmd_batch(xb)
            optimizer.zero_grad()
            out, _, _ = model(xb)
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
                out, _, _ = model(xb)
                v_preds.extend(out.cpu().numpy().flatten())
                v_trues.extend(yb.numpy().flatten())

        m = compute_metrics(np.array(v_trues), np.array(v_preds))
        val_mae, val_pcc = m["MAE"], m["PCC"]
        val_score = val_mae + args.alpha_pcc * (1.0 - val_pcc)

        if val_score < best_score:
            best_score = val_score
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            stat = f"[BEST -> MAE: {val_mae:.3f} | PCC: {val_pcc:.3f} | Acc@2bpm: {m['Acc@2bpm(%)']:.1f}%]"
        else:
            patience_cnt += 1
            stat = f"[Patience: {patience_cnt}/{patience}] (PCC: {val_pcc:.3f})"

        if epoch % 5 == 0 or epoch == 1 or patience_cnt == 0:
            print(f"Epoch [{epoch:03d}/{epochs}] - Loss: {train_loss/len(train_loader.dataset):.4f} - Val MAE: {val_mae:.3f} {stat}")

        if patience_cnt >= patience:
            print(f"\n[INFO] Early stopping pada epoch {epoch}.")
            break

    print(f"\nPelatihan Teacher selesai. Checkpoint tersimpan di: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    run_efficiency_report(model, device=cfg.train.device, simulate_wearable=cfg.SIMULATE_WEARABLE)

if __name__ == "__main__":
    main()