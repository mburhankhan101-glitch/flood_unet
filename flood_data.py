"""
Loading Sen1Floods11 hand-labelled chips for training and evaluation.

Radar (VV, VH, dB) is clipped to [-35, 5] dB, then standardised with the mean and spread of the
TRAINING chips only. Missing radar becomes 0 (the average) and its pixels get label -1, so neither
the loss nor the scores use them: water can't be mapped where the radar saw nothing.
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio

DATA = "data"
CLIP = (-35.0, 5.0)


def split_chips(split):
    df = pd.read_csv(os.path.join(DATA, "splits", f"{split}.csv"), header=None)
    return [f.replace("_S1Hand.tif", "") for f in df[0]]


def read(folder, chip):
    with rasterio.open(os.path.join(DATA, folder, f"{chip}_{folder}.tif")) as src:
        return src.read()


def raw_chip(chip):
    """Radar (2, 512, 512) clipped dB with NaN where missing, and labels (512, 512): 1 water, 0 not, -1 ignore."""
    s1 = read("S1Hand", chip).astype("float32")
    label = read("LabelHand", chip)[0].astype("int64")
    missing = ~np.isfinite(s1).all(axis=0)
    s1 = np.clip(s1, *CLIP)
    s1[:, missing] = np.nan
    label[missing] = -1
    return s1, label


def norm_stats(chips, path):
    """Per-band mean and standard deviation over the training chips (cached as JSON)."""
    if os.path.exists(path):
        return json.load(open(path))
    vals = [[], []]
    for chip in chips:
        s1, _ = raw_chip(chip)
        for b in range(2):
            v = s1[b][np.isfinite(s1[b])]
            vals[b].append(v[::7])          # a sample is plenty for a mean and a spread
    stats = {"mean": [float(np.mean(np.concatenate(v))) for v in vals],
             "std": [float(np.std(np.concatenate(v))) for v in vals]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(stats, open(path, "w"), indent=1)
    return stats


def load_split(split, stats):
    """All chips of a split in memory: x (n, 2, 512, 512) float32 standardised, y (n, 512, 512) int64."""
    chips = split_chips(split)
    xs, ys = [], []
    mean = np.array(stats["mean"], dtype="float32")[:, None, None]
    std = np.array(stats["std"], dtype="float32")[:, None, None]
    for chip in chips:
        s1, label = raw_chip(chip)
        x = (s1 - mean) / std
        xs.append(np.nan_to_num(x, nan=0.0))
        ys.append(label)
    return chips, np.stack(xs), np.stack(ys)


def pooled_iou(pred, label):
    """IoU of water summed over all pixels that have a label: TP / (TP + FP + FN)."""
    ok = label >= 0
    tp = int(((pred == 1) & (label == 1) & ok).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred != 1) & (label == 1)).sum())
    return tp / max(tp + fp + fn, 1), tp, fp, fn
