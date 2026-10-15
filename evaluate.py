"""
Score a trained U-Net against the hand labels, next to the dataset's Otsu baseline, on the same pixels.
The input bands (s1, s2 or s1s2) are read from the run's args.json.

Test sets: test (90 chips, the training events), Bolivia (15 chips, an event never seen) and the
Pakistan test chips (6). Pooled IoU of water, plus per-chip results, and a figure of the Pakistan
test chips: radar, hand label, Otsu, U-Net.

The probability threshold that turns the U-Net's output into water / not water is chosen on the
validation chips (0.05 to 0.95) and then applied unchanged to test, Bolivia and Pakistan; results at
0.5 are reported too.

For a radar + optical model there is one more check: test and Bolivia again with the optical bands
blanked (set to their training average, i.e. no information). Clouds are mostly left unlabelled in
the hand labels, so the normal test says little about a cloudy day; this shows whether the model can
still fall back on radar.

    python evaluate.py                  # runs/s1_unet/best.pt
    python evaluate.py --name s1_unet
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from flood_data import band_stats, load_split, pooled_iou, raw_chip, raw_s2, read, split_chips
from train import predict
from unet import UNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="s1_unet")
    p.add_argument("--base", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    run = os.path.join("runs", args.name)
    bands = json.load(open(os.path.join(run, "args.json"))).get("bands", "s1")
    stats = band_stats(bands)
    model = UNet(cin=sum(len(st["mean"]) for st in stats.values()), base=args.base)
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"), weights_only=True))
    rows, chip_rows = [], []
    figure = None
    _, xv, yv = load_split("valid", stats, bands)
    pv = predict(model, xv)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = [pooled_iou((pv > t).astype(int), yv)[0] for t in grid]
    thr = float(grid[int(np.argmax(scores))])
    print(f"threshold chosen on validation: {thr} (validation IoU {max(scores):.3f}; at 0.5: "
          f"{pooled_iou((pv > 0.5).astype(int), yv)[0]:.3f})")
    for split in ("valid", "test", "bolivia"):
        chips, x, y = (_, xv, yv) if split == "valid" else load_split(split, stats, bands)
        if split == "valid":
            chips = split_chips("valid")
        prob = pv if split == "valid" else predict(model, x)
        pred = (prob > thr).astype(int)
        otsu = np.stack([read("S1OtsuLabelHand", c)[0] for c in chips])
        groups = [(split, np.ones(len(chips), bool))]
        blanked = None
        if bands == "s1s2" and split != "valid":
            xb = x.copy()
            xb[:, 2:] = 0
            blanked = (predict(model, xb) > thr).astype(int)
            del xb
        if split == "test":
            groups.append(("Pakistan (test chips)", np.array([c.startswith("Pakistan") for c in chips])))
            pk = np.flatnonzero(groups[-1][1])
            figure = ([chips[i] for i in pk], y[pk], otsu[pk], prob[pk])
        for name, m in groups:
            methods = [("Otsu", otsu[m]), ("U-Net at 0.5", (prob[m] > 0.5).astype(int)),
                       (f"U-Net at {thr} (tuned)", pred[m])]
            if blanked is not None:
                methods.append((f"U-Net at {thr}, optical removed", blanked[m]))
            for method, p in methods:
                iou, tp, fp, fn = pooled_iou(p, y[m])
                rows.append({"set": name, "method": method, "chips": int(m.sum()), "iou": round(iou, 3),
                             "found": tp, "false_alarms": fp, "missed": fn})
        for i, c in enumerate(chips):
            chip_rows.append({"split": split, "chip": c, "otsu_iou": round(pooled_iou(otsu[i], y[i])[0], 3),
                              "unet_iou": round(pooled_iou(pred[i], y[i])[0], 3)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(run, "results.csv"), index=False)
    pd.DataFrame(chip_rows).to_csv(os.path.join(run, "chip_results.csv"), index=False)
    print(res.to_string(index=False))

    chips, y, otsu, prob = figure
    optical = "s2" in bands
    fig, axes = plt.subplots(len(chips), 5 if optical else 4, figsize=(13 if optical else 10.5, 2.7 * len(chips)))
    for r, chip in enumerate(chips):
        lab = np.where(y[r] < 0, np.nan, y[r])
        panels = [(raw_chip(chip)[0][0], "gray", (-25, 0), f"{chip}: radar VV (dB)")]
        if optical:
            rgb = np.nan_to_num(raw_s2(chip)[[3, 2, 1]].transpose(1, 2, 0) / 3000).clip(0, 1)
            panels.append((rgb, None, (0, 1), "optical, true colour"))
        panels += [(lab, "Blues", (0, 1), "hand label"),
                  (np.where(y[r] < 0, np.nan, otsu[r]), "Blues", (0, 1), f"Otsu, IoU {pooled_iou(otsu[r], y[r])[0]:.2f}"),
                  (np.where(y[r] < 0, np.nan, prob[r] > thr), "Blues", (0, 1),
                   f"U-Net, IoU {pooled_iou((prob[r] > thr).astype(int), y[r])[0]:.2f}")]
        for c, (img, cmap, lim, title) in enumerate(panels):
            axes[r, c].imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
            axes[r, c].set_title(title, fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(run, "pakistan_test.png"), dpi=100)
    plt.close(fig)
    print(f"Saved results.csv, chip_results.csv, pakistan_test.png in {run}")


if __name__ == "__main__":
    main()
