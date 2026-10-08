"""
Evaluate pretrained MfamaNet weights on user's NewHandPD data.
Usage: python evaluate.py --task meander
"""
import argparse
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)

from models import MfamaNet
from dataset import ImageSignalDataset
from torch.utils.data import DataLoader

MODEL_CONFIG = dict(
    num_classes=2, num_hiddens=96, dropout=0.3,
    signal_chans=6, scale_lst=[625, 1250, 2500, 5000, 6250],
    num_layers=[2, 4, 6, 8, 8],
    image_chans=3, num_blocks=[2, 4, 4, 6, 6],
    scale_kernels=[3, 5, 7, 11, 17], drop_path_rate=0.1,
    tre_type='token', use_mfa=False,
)


def load_model(task, device):
    model = MfamaNet(**MODEL_CONFIG)
    w = os.path.join('params', f'{task}.pth')
    state = torch.load(w, map_location='cpu')
    model.load_state_dict(state, strict=False)   # RoPE buffer names differ after our speedup fix
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    for img, sig, label in loader:
        img = img.to(device)
        sig = sig.to(device)
        logits, _ = model(img, sig)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        all_labels.extend(label.numpy().tolist())
        all_preds.extend(pred.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(classification_report(all_labels, all_preds,
                                 target_names=['Healthy', 'Patient'],
                                 zero_division=0))
    return acc, auc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['meander', 'spiral', 'circle'])
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    csv = f'{args.task}.csv'
    if not os.path.exists(csv):
        raise SystemExit(f'{csv} not found')

    ds = ImageSignalDataset(csv, root='.')
    print(f'{args.task}: {len(ds)} samples '
          f'(healthy={sum(1 for l in ds.labels if l==0)}, '
          f'patient={sum(1 for l in ds.labels if l==1)})')

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = load_model(args.task, device)
    evaluate(model, loader, device)
