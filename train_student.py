import os
import argparse
import torch
import numpy as np
from config import get_config
from src.dataset import get_dataloaders
from src.models.transformer import build_transformer
from src.losses import CorrelationAwareLoss, KnowledgeDistillationLoss
from src.efficiency import compute_metrics, run_efficiency_report
from src.vmd_utils import apply_vmd_batch

def main():
    parser = argparse.ArgumentParser(description="Pelatihan Model Student (KD / SCRATCH)")
    parser.add_argument("--mode", type=str, default="KD", choices=["KD", "SCRATCH"])
    parser.add_argument("--alpha_pcc", type=float, default=2.0)
    parser.add_argument("--use_vmd", action="store_true", default=False, help="Aktifkan VMD pada input sinyal")
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.train.device)
    use_vmd = args.use_vmd or getattr(cfg, "USE_VMD", False)

    print("=" * 65)
    print(f" PELATIHAN STUDENT ({args.mode}) (VMD: {use_vmd})")
    print(f" Device: {device} | Dataset: {cfg.ACTIVE_DATASET}")
    print("=" * 65)

    torch.manual_seed(cfg.split.random_seed)
    np.random.seed(cfg.split.random_seed)

    train_loader, val_loader, _ = get_dataloaders(cfg, dataset_name=cfg.ACTIVE_DATASET)
    student = build_transformer(cfg.student_model, is_student=True).to(device)

    teacher = None
    if args.mode == "KD":
        teacher = build_transformer(cfg.teacher_model, is_student=False).to(device)
        t_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "teacher_best.pth")
        if not os.path.exists(t_ckpt):
            raise FileNotFoundError(f"Checkpoint teacher {t_ckpt} tidak ditemukan. Jalankan train_teacher.py terlebih dahulu.")
        teacher.load_state_dict(torch.load(t_ckpt, map_location=device, weights_only=False))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        criterion = KnowledgeDistillationLoss(alpha_kd=cfg.kd.alpha, alpha_pcc=args.alpha_pcc)
    else:
        criterion = CorrelationAwareLoss(alpha_pcc=args.alpha_pcc)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)

    best_score = float("inf")
    patience = cfg.train.patience
    patience_cnt = 0
    save_filename = "student_kd_best.pth" if args.mode == "KD" else "student_scratch_best.pth"
    save_path = os.path.join(cfg.CHECKPOINT_DIR, save_filename)

    for epoch in range(1, cfg.train.epochs + 1):
        student.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_vmd:
                xb, _ = apply_vmd_batch(xb)
            optimizer.zero_grad()

            s_out, _, s_feat = student(xb)

            if args.mode == "KD":
                with torch.no_grad():
                    t_out, _, t_feat = teacher(xb)
                loss = criterion(s_out, t_out, yb, student_feat=s_feat, teacher_feat=t_feat)
            else:
                loss = criterion(s_out, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)

        scheduler.step()

        # Validasi
        student.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                if use_vmd:
                    xb, _ = apply_vmd_batch(xb)
                s_out, _, _ = student(xb)
                v_preds.extend(s_out.cpu().numpy().flatten())
                v_trues.extend(yb.numpy().flatten())

        m = compute_metrics(np.array(v_trues), np.array(v_preds))
        val_mae, val_pcc = m["MAE"], m["PCC"]
        val_score = val_mae + args.alpha_pcc * (1.0 - val_pcc)

        if val_score < best_score:
            best_score = val_score
            patience_cnt = 0
            torch.save(student.state_dict(), save_path)
            stat = f"[BEST -> MAE: {val_mae:.3f} | PCC: {val_pcc:.3f} | Acc@2bpm: {m['Acc@2bpm(%)']:.1f}%]"
        else:
            patience_cnt += 1
            stat = f"[Patience: {patience_cnt}/{patience}] (PCC: {val_pcc:.3f})"

        if epoch % 5 == 0 or epoch == 1 or patience_cnt == 0:
            print(f"Epoch [{epoch:03d}/{cfg.train.epochs}] - Loss: {train_loss/len(train_loader.dataset):.4f} - Val MAE: {val_mae:.3f} {stat}")

        if patience_cnt >= patience:
            print(f"\n[INFO] Early stopping pada epoch {epoch}.")
            break

    print(f"\nPelatihan Student ({args.mode}) selesai. Checkpoint tersimpan di: {save_path}")
    student.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    run_efficiency_report(student, device=cfg.train.device, simulate_wearable=cfg.SIMULATE_WEARABLE)

if __name__ == "__main__":
    main()