"""
Train / fine-tune MfamaNet on NewHandPD.
Official repo does NOT provide training code; this is a from-scratch trainer.

Usage:
  # Fine-tune from pretrained weights (recommended for small dataset)
  python train.py --task meander --epochs 50 --batch-size 8 --lr 1e-4 --pretrained

  # Train from scratch
  python train.py --task meander --epochs 100 --batch-size 8 --lr 5e-4
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe in scripts
import matplotlib.pyplot as plt

from models import MfamaNet
from dataset import get_dataloaders


def plot_curves(history, save_path, task):
    """Draw training loss + accuracy/AUC convergence curves (2 subplots)."""
    ep = history['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Loss ---
    ax = axes[0]
    ax.plot(ep, history['train_loss'], '-o', ms=3, lw=1.5,
            color='#d62728', label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{task} — Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Accuracy / AUC ---
    ax = axes[1]
    ax.plot(ep, history['train_acc'], '-o', ms=3, lw=1.5,
            color='#1f77b4', label='Train Acc')
    ax.plot(ep, history['test_acc'], '-s', ms=3, lw=1.5,
            color='#2ca02c', label='Test Acc')
    ax.plot(ep, history['test_auc'], '-^', ms=3, lw=1.5,
            color='#ff7f0e', label='Test AUC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(f'{task} — Accuracy & AUC')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Curve saved: {save_path}')


def build_model(use_mfa, mfa_weight):
    cfg = dict(
        num_classes=2, num_hiddens=96, dropout=0.3,
        signal_chans=6, scale_lst=[625, 1250, 2500, 5000, 6250],
        num_layers=[2, 4, 6, 8, 8],
        image_chans=3, num_blocks=[2, 4, 4, 6, 6],
        scale_kernels=[3, 5, 7, 11, 17], drop_path_rate=0.1,
        tre_type='token', use_mfa=use_mfa, mfa_weight=mfa_weight,
    )
    return MfamaNet(**cfg)


def train_one_epoch(model, loader, opt, device, accum_steps=1, use_amp=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    opt.zero_grad()
    for step, (img, sig, label) in enumerate(loader):
        img, sig, label = img.to(device), sig.to(device), label.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=use_amp and device.type == 'cuda'):
            logits, mfa_loss = model(img, sig)
            ce = F.cross_entropy(logits, label)
            loss = (ce + mfa_loss) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        total_loss += loss.item() * accum_steps * label.size(0)
        correct += (logits.argmax(-1) == label).sum().item()
        total += label.size(0)
    # leftover gradients
    if (step + 1) % accum_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, device, use_amp=True):
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    for img, sig, label in loader:
        img, sig = img.to(device), sig.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=use_amp and device.type == 'cuda'):
            logits, _ = model(img, sig)
        probs = F.softmax(logits.float(), dim=-1)
        correct += (logits.argmax(-1) == label.to(device)).sum().item()
        total += label.size(0)
        all_probs.extend(probs[:, 1].cpu().tolist())
        all_labels.extend(label.tolist())
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    return correct / total, auc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['meander', 'spiral', 'circle'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--accum-steps', type=int, default=8,
                    help='Gradient accumulation steps (effective batch = batch_size*accum)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--pretrained', action='store_true',
                    help='Load pretrained weights (use_mfa=False in official ckpt)')
    ap.add_argument('--use-mfa', action='store_true',
                    help='Enable MFA loss (only works from scratch or with strict=False)')
    ap.add_argument('--mfa-weight', type=float, default=0.6)
    ap.add_argument('--out', default='checkpoints')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    train_loader, test_loader = get_dataloaders(
        f'{args.task}.csv', batch_size=args.batch_size)
    print(f'Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}')

    model = build_model(args.use_mfa, args.mfa_weight).to(device)

    if args.pretrained:
        w = os.path.join('params', f'{args.task}.pth')
        state = torch.load(w, map_location='cpu')
        # If use_mfa=True, the official ckpt has no mfa_* params -> strict=False
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'Pretrained load: {len(missing)} missing keys (expected if --use-mfa)')
        if unexpected:
            print(f'Pretrained load: {len(unexpected)} unexpected keys')

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [],
               'test_acc': [], 'test_auc': []}
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device,
                                          accum_steps=args.accum_steps)
        te_acc, te_auc = eval_model(model, test_loader, device)
        sched.step()
        print(f'Epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f}  '
              f'train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  test_auc={te_auc:.4f}')

        history['epoch'].append(ep)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['test_auc'].append(te_auc)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(),
                       os.path.join(args.out, f'{args.task}_best.pth'))

        # Save curve after every epoch so it's viewable mid-training
        plot_curves(history,
                    os.path.join(args.out, f'{args.task}_curve.png'),
                    args.task)

    # Save raw history
    pd.DataFrame(history).to_csv(
        os.path.join(args.out, f'{args.task}_history.csv'), index=False)
    print(f'Best test accuracy: {best_acc:.4f}')
