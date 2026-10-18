"""
Loading Sen1Floods11 hand-labelled chips for training and evaluation.

Radar (VV, VH, dB) is clipped to [-35, 5] dB, then standardised with the mean and spread of the
TRAINING chips only. Missing radar becomes 0 (the average) and its pixels get label -1, so neither
the loss nor the scores use them: water can't be mapped where the radar saw nothing.

Optical (Sentinel-2, 13 bands, reflectance x 10000) is standardised the same way, band by band.
Missing optical (0 in every band) also becomes 0, but the label is kept: it is only 0.05-0.66% of the
labelled pixels per split, and this way every model is scored on exactly the same pixels.

--bands picks the inputs: s1 (VV, VH), s2 (13 optical bands), s1s2 (all 15, radar first) or s1pre
(VV, VH on the flood day, then VV, VH before it, from data/S1PreHand made by s1_pre.py). The before
image is standardised with the same radar statistics as the flood day, so differences keep their
meaning; a chip without a before image gets 0 there (unknown).
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio

DATA = "data"
CLIP = (-35.0, 5.0)
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
SENSORS = {"s1": ["s1"], "s2": ["s2"], "s1s2": ["s1", "s2"], "s1pre": ["s1", "pre"]}
CHANNELS = {"s1": 2, "s2": 13, "s1s2": 15, "s1pre": 4}


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


def raw_s2(chip):
    """Optical (13, 512, 512) reflectance x 10000, NaN where the chip has no optical data."""
    s2 = read("S2Hand", chip).astype("float32")
    s2[:, (s2 == 0).all(axis=0) | ~np.isfinite(s2).all(axis=0)] = np.nan
    return s2


def raw_pre(chip):
    """Before-flood radar (2, 512, 512) clipped dB, NaN where there is none."""
    path = os.path.join(DATA, "S1PreHand", f"{chip}_S1PreHand.tif")
    if not os.path.exists(path):
        return np.full((2, 512, 512), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        pre = src.read().astype("float32")
    missing = ~np.isfinite(pre).all(axis=0)
    pre = np.clip(pre, *CLIP)
    pre[:, missing] = np.nan
    return pre


def norm_stats(chips, path, sensor="s1"):
    """Per-band mean and standard deviation over the training chips (cached as JSON)."""
    if os.path.exists(path):
        return json.load(open(path))
    reader = (lambda c: raw_chip(c)[0]) if sensor == "s1" else raw_s2
    vals = None
    for chip in chips:
        img = reader(chip)
        vals = vals or [[] for _ in img]
        for b in range(len(img)):
            v = img[b][np.isfinite(img[b])]
            vals[b].append(v[::7])          # a sample is plenty for a mean and a spread
    stats = {"mean": [float(np.mean(np.concatenate(v))) for v in vals],
             "std": [float(np.std(np.concatenate(v))) for v in vals]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(stats, open(path, "w"), indent=1)
    return stats


def band_stats(bands="s1"):
    """Training-chip statistics for each sensor in bands: {"s1": {...}, "s2": {...}}."""
    train = split_chips("train")
    sensors = {"s1" if s == "pre" else s for s in SENSORS[bands]}      # before uses the radar statistics
    return {s: norm_stats(train, os.path.join("runs", f"norm_{s}.json"), s) for s in sensors}


def load_split(split, stats, bands="s1"):
    """All chips of a split in memory: x (n, channels, 512, 512) standardised, y (n, 512, 512) int64.

    x is float32 for radar alone and float16 with optical bands (15 bands of 512 x 512 for 341 chips
    would not fit in a laptop's memory as float32); batches are turned back into float32."""
    chips = split_chips(split)
    xs, ys = [], []
    dtype = "float32" if bands in ("s1", "s1pre") else "float16"
    for chip in chips:
        s1, label = raw_chip(chip)
        parts = []
        for sensor in SENSORS[bands]:
            img = s1 if sensor == "s1" else raw_pre(chip) if sensor == "pre" else raw_s2(chip)
            st = stats["s1" if sensor == "pre" else sensor]
            mean = np.array(st["mean"], dtype="float32")[:, None, None]
            std = np.array(st["std"], dtype="float32")[:, None, None]
            parts.append(np.nan_to_num((img - mean) / std, nan=0.0))
        xs.append(np.concatenate(parts).astype(dtype))
        ys.append(label)
    return chips, np.stack(xs), np.stack(ys)


def pooled_iou(pred, label):
    """IoU of water summed over all pixels that have a label: TP / (TP + FP + FN)."""
    ok = label >= 0
    tp = int(((pred == 1) & (label == 1) & ok).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred != 1) & (label == 1)).sum())
    return tp / max(tp + fp + fn, 1), tp, fp, fn
