"""
A first look at the hand-labelled Sen1Floods11 chips, and the score to beat.

- How many chips per split and event, how much of each chip is water, how much has no label.
- Radar values (VV, VH in dB) over water and over land: why a threshold works at all.
- The baseline: the dataset's own Otsu-threshold water map (S1OtsuLabelHand) scored against the hand
  labels with IoU, per split and for the Pakistan chips.
- A figure of example chips: radar, true colour, hand label, Otsu map.

    python explore.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

DATA = "data"
OUT = "runs"


def read(folder, chip):
    with rasterio.open(os.path.join(DATA, folder, f"{chip}_{folder}.tif")) as src:
        return src.read()


def split_chips(split):
    df = pd.read_csv(os.path.join(DATA, "splits", f"{split}.csv"), header=None)
    return [f.replace("_S1Hand.tif", "") for f in df[0]]


def iou(pred, label):
    """Intersection over union of the water class, over pixels that have a label."""
    ok = label >= 0
    p, t = pred[ok] == 1, label[ok] == 1
    union = (p | t).sum()
    return (p & t).sum() / union if union else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    vv_water, vv_land = [], []
    for split in ("train", "valid", "test", "bolivia"):
        for chip in split_chips(split):
            label = read("LabelHand", chip)[0]
            otsu = read("S1OtsuLabelHand", chip)[0]
            s1 = read("S1Hand", chip)
            ok = label >= 0
            rows.append({"split": split, "chip": chip, "event": chip.split("_")[0],
                         "nodata": float(np.mean(~ok)), "water": float(np.mean(label[ok] == 1)) if ok.any() else np.nan,
                         "tp": int(((otsu == 1) & (label == 1) & ok).sum()), "fp": int(((otsu == 1) & (label == 0)).sum()),
                         "fn": int(((otsu != 1) & (label == 1)).sum())})
            if split == "train":
                vv = s1[0]
                good = ok & np.isfinite(vv)
                vv_water.append(vv[good & (label == 1)][::50])
                vv_land.append(vv[good & (label == 0)][::50])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "chips.csv"), index=False)

    print(df.groupby("split").agg(chips=("chip", "size"), water=("water", "mean"), nodata=("nodata", "mean")).round(3))
    print("\nOtsu baseline vs hand labels (IoU of water, pooled over all pixels of the split):")
    for name, g in list(df.groupby("split")) + [("Pakistan (all splits)", df[df["event"] == "Pakistan"])]:
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        print(f"  {name:<22} IoU {tp / (tp + fp + fn):.3f}   (chips {len(g)})")

    w, l = np.concatenate(vv_water), np.concatenate(vv_land)
    print(f"\nVV in dB (training chips): water median {np.median(w):.1f}, land median {np.median(l):.1f}")
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.linspace(-35, 5, 81)
    ax.hist(l, bins=bins, alpha=0.6, density=True, label="not water")
    ax.hist(w, bins=bins, alpha=0.6, density=True, label="water")
    ax.set_xlabel("VV backscatter (dB)")
    ax.set_ylabel("share of pixels")
    ax.legend()
    ax.set_title("Why radar can see floods: water is dark")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vv_histogram.png"), dpi=130)
    plt.close(fig)

    pk = df[df["event"] == "Pakistan"].sort_values("water", ascending=False).head(4)["chip"]
    fig, axes = plt.subplots(len(pk), 4, figsize=(11, 2.8 * len(pk)))
    for r, chip in enumerate(pk):
        s1, s2 = read("S1Hand", chip), read("S2Hand", chip)
        rgb = np.clip(s2[[3, 2, 1]].transpose(1, 2, 0) / 3000, 0, 1)
        label, otsu = read("LabelHand", chip)[0], read("S1OtsuLabelHand", chip)[0]
        panels = [(s1[0], "gray", (-25, 0), "radar VV"), (rgb, None, None, "true colour"),
                  (np.where(label < 0, np.nan, label), "Blues", (0, 1), "hand label"),
                  (np.where(label < 0, np.nan, otsu), "Blues", (0, 1), f"Otsu, IoU {iou(otsu, label):.2f}")]
        for c, (img, cmap, lim, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            ax.set_title(f"{chip}: {title}" if c == 0 else title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pakistan_examples.png"), dpi=110)
    plt.close(fig)
    print(f"Saved chips.csv, vv_histogram.png, pakistan_examples.png in {OUT}")


if __name__ == "__main__":
    main()
