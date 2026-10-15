"""
Train a U-Net to map water from Sentinel-1 (VV, VH), Sentinel-2 (13 bands) or both (--bands), with
early stopping on the validation split.

Each epoch: one random 256 x 256 crop per training chip (252 crops), mini-batches of --batch, Adam,
cross-entropy with class weights from the training labels (water is rare) and -1 pixels ignored.
After each epoch the validation chips are mapped whole and scored with pooled IoU; the best epoch is
kept (best.pt). Training stops when validation IoU has not improved for --patience epochs.

--augment adds a random 90-degree rotation, a random flip and a random brightness shift of up to
--jitter dB (the same shift on VV and VH; optical bands are not shifted) to every training crop; the label is rotated and flipped
with it. --loss ce+dice adds a soft Dice loss on the water class to the weighted cross-entropy.

A checkpoint (last.pt) is written after every epoch, so a run that is stopped (--budget minutes, a
crash, a closed laptop) continues where it left off when started again.

    python train.py                  # runs/s1_unet/
    python train.py --name test --epochs 2
    python train.py --name s1s2_unet --bands s1s2 --patience 20
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from flood_data import band_stats, load_split, pooled_iou
from unet import UNet

SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="s1_unet")
    p.add_argument("--bands", choices=["s1", "s2", "s1s2"], default="s1")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=16, help="filters in the first layer")
    p.add_argument("--augment", action="store_true", help="rotations, flips and dB jitter")
    p.add_argument("--jitter", type=float, default=1.0, help="largest brightness shift in dB (with --augment)")
    p.add_argument("--loss", choices=["ce", "ce+dice"], default="ce")
    p.add_argument("--budget", type=float, default=9.0, help="minutes; stop after the epoch that passes this")
    return p.parse_args()


def augment(x, y, rng, jitter_norm):
    """The same random rotation and flip for image and label; one brightness shift for the crop."""
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    x = x + rng.uniform(-1, 1) * jitter_norm[:, None, None]    # dB shift, in standardised units per band
    return np.ascontiguousarray(x, dtype="float32"), np.ascontiguousarray(y)


def dice_loss(logits, y):
    """1 - soft Dice of the water class over the labelled pixels of the batch."""
    p = torch.softmax(logits, 1)[:, 1]
    ok = (y >= 0).float()
    t = (y == 1).float()
    inter = (p * t * ok).sum()
    return 1 - (2 * inter + 1) / ((p * ok).sum() + (t * ok).sum() + 1)


def predict(model, x, batch=4):
    """Water probability for whole chips (n, 2, 512, 512) -> (n, 512, 512)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(torch.softmax(model(torch.from_numpy(x[i:i + batch]).float()), 1)[:, 1].numpy())
    return np.concatenate(out)


def main():
    args = parse_args()
    run = os.path.join("runs", args.name)
    os.makedirs(run, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    stats = band_stats(args.bands)
    _, xtr, ytr = load_split("train", stats, args.bands)
    _, xva, yva = load_split("valid", stats, args.bands)
    n_land, n_water = int((ytr == 0).sum()), int((ytr == 1).sum())
    weights = torch.tensor([1.0, n_land / n_water], dtype=torch.float32)
    print(f"{args.bands}: {xtr.shape[1]} input bands; train {len(xtr)} chips, valid {len(xva)}; water {n_water / (n_land + n_water):.1%} of labelled "
          f"training pixels -> water weight {weights[1]:.1f}", flush=True)

    torch.manual_seed(SEED)
    model = UNet(cin=xtr.shape[1], base=args.base)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    loss_fn = (lambda out, y: ce(out, y) + dice_loss(out, y)) if args.loss == "ce+dice" else ce
    jitter_norm = np.zeros(xtr.shape[1], dtype="float32")        # dB shift on the radar bands only
    if "s1" in stats:
        jitter_norm[:2] = args.jitter / np.array(stats["s1"]["std"], dtype="float32")
    rng = np.random.default_rng(SEED)
    state = {"epoch": 0, "best": -1.0, "wait": 0, "history": []}
    last = os.path.join(run, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"resuming after epoch {state['epoch']} (best validation IoU {state['best']:.3f})", flush=True)
    json.dump(vars(args), open(os.path.join(run, "args.json"), "w"), indent=1)
    print(f"U-Net: {sum(p.numel() for p in model.parameters()):,} weights", flush=True)

    c = args.crop
    while state["epoch"] < args.epochs and state["wait"] < args.patience:
        te = time.time()
        model.train()
        order = rng.permutation(len(xtr))
        total, batches = 0.0, 0
        for i in range(0, len(order), args.batch):
            xb, yb = [], []
            for k in order[i:i + args.batch]:
                for _ in range(10):  # a crop with at least some labelled pixels
                    r, s = rng.integers(0, 512 - c + 1, size=2)
                    if (ytr[k, r:r + c, s:s + c] >= 0).mean() > 0.2:
                        break
                xc, yc = xtr[k, :, r:r + c, s:s + c], ytr[k, r:r + c, s:s + c]
                if args.augment:
                    xc, yc = augment(xc, yc, rng, jitter_norm)
                xb.append(xc)
                yb.append(yc)
            loss = loss_fn(model(torch.from_numpy(np.stack(xb)).float()), torch.from_numpy(np.stack(yb)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        p = predict(model, xva)
        iou, *_ = pooled_iou((p > 0.5).astype(int), yva)
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": round(total / batches, 4),
                                 "valid_iou": round(iou, 4), "seconds": round(time.time() - te, 1)})
        if iou > state["best"] + 1e-4:
            state["best"], state["wait"] = iou, 0
            torch.save(model.state_dict(), os.path.join(run, "best.pt"))
        else:
            state["wait"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, last)
        pd.DataFrame(state["history"]).to_csv(os.path.join(run, "history.csv"), index=False)
        print(f"  epoch {state['epoch']:>2}: loss {total / batches:.3f}, validation IoU {iou:.3f}"
              f"{' (best)' if state['wait'] == 0 else ''}, {time.time() - te:.0f} s", flush=True)
        if (time.time() - t0) / 60 > args.budget:
            print("Time budget reached: run again to continue.")
            return
    print(f"Finished: best validation IoU {state['best']:.3f} after {state['epoch']} epochs "
          f"({'early stop' if state['wait'] >= args.patience else 'epoch limit'})")


if __name__ == "__main__":
    main()
