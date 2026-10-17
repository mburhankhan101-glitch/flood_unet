"""
Cloud masking with s2cloudless, tested with real clouds pasted onto clear chips.

A cloud is bright in optical and looks like dry land, so a model that trusts optical calls the water
under it "land". The fix: detect clouds with s2cloudless (Sentinel Hub's per-pixel cloud detector),
set the optical bands under the mask to 0 (the training average, i.e. missing) and let the sensor
dropout model (runs/s1s2_drop50) fall back on radar there.

The hand labels mostly leave clouds unlabelled, so models can't be scored under real clouds. Instead,
real clouds are cut from training chips and pasted onto validation and test chips, whose labels were
drawn on clear imagery and stay true under the pasted cloud. Radar is left unchanged (it sees through
clouds). The clouds to paste are found WITHOUT s2cloudless: bright (mean of B2-B4 above 0.2), left
unlabelled by the annotators, with both sensors present. Otherwise every pasted cloud would be one the
mask is known to find. The edges are softened so thin cloud borders are tested too.

The mask's probability cut is chosen on the validation chips (by flood IoU of the dropout model with
the mask) and then used unchanged on test. Smoothing (4-pixel disk) and dilation (2 pixels) are fixed
in advance. Cloud shadows are not pasted, so this test is easier than reality.

    python clouds.py          # writes runs/clouds/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from s2cloudless import S2PixelCloudDetector
from scipy import ndimage

from flood_data import band_stats, load_split, pooled_iou, read, split_chips
from train import predict
from unet import UNet

OUT = os.path.join("runs", "clouds")
CACHE = os.path.join("cache", "clouds")
SMOOTH, DILATE = 4, 2                  # pixels, fixed in advance
GRID = [0.3, 0.4, 0.5, 0.6, 0.7]       # cloud-probability cuts tried on validation
DETECTOR = S2PixelCloudDetector(all_bands=True, average_over=SMOOTH, dilation_size=DILATE)


def cloud_sources():
    """Training chips with real clouds: bright, unlabelled, both sensors present; 10-80% of the chip."""
    out = []
    for chip in split_chips("train"):
        s2 = read("S2Hand", chip).astype("float32") / 10000
        stencil = ((read("LabelHand", chip)[0] == -1) & np.isfinite(read("S1Hand", chip)).all(0)
                   & (s2 > 0).all(0) & (s2[1:4].mean(0) > 0.2))
        stencil = ndimage.binary_closing(ndimage.binary_opening(stencil, iterations=2), iterations=3)
        if 0.1 <= stencil.mean() <= 0.8:
            out.append((chip, stencil))
    return out


def paste(clear, cloud, stencil):
    """Optical (13, 512, 512) of a clear chip with a cloud blended in; returns it and the cloud cover (0-1)."""
    cover = ndimage.gaussian_filter(stencil.astype("float32"), 2).clip(0, 1)
    cover[~(cloud > 0).all(0) | ~(clear > 0).all(0)] = 0
    return cover * cloud + (1 - cover) * clear, cover


def cloud_prob(s2, key):
    """s2cloudless probability for one chip (reflectance x 10000), cached on disk."""
    path = os.path.join(CACHE, f"{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    p = DETECTOR.get_cloud_probability_maps((s2 / 10000).transpose(1, 2, 0)[None])[0]
    p[~(s2 > 0).all(0)] = 0
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, p.astype("float16"))
    return p.astype("float16")


def mask_from(prob, cut):
    return DETECTOR.get_mask_from_prob(prob[None].astype("float32"), cut)[0].astype(bool)


def standardise(s2, st):
    m = np.array(st["mean"], "float32")[:, None, None]
    s = np.array(st["std"], "float32")[:, None, None]
    x = (s2 - m) / s
    x[:, ~(s2 > 0).all(0)] = 0
    return x


def load_model(name, cin):
    model = UNet(cin=cin)
    model.load_state_dict(torch.load(os.path.join("runs", name, "best.pt"), weights_only=True))
    res = pd.read_csv(os.path.join("runs", name, "results.csv"))
    cut = float(res[(res.set == "valid") & res.method.str.endswith("(tuned)")].method.iloc[0].split()[2])
    return model, cut


def build(split, stats, sources, real=True):
    """Inputs for one split: clear (n, 15), clouded (n, 15), cloud cover, and cloud probabilities."""
    chips, x, y = load_split(split, stats, "s1s2")
    rng = np.random.default_rng(0)
    xc = x.copy()
    cover = np.zeros(y.shape, "float16")
    p_real = np.zeros(y.shape, "float16")
    p_cloud = np.zeros(y.shape, "float16")
    for i, chip in enumerate(chips):
        src, stencil = sources[rng.integers(len(sources))]
        clear = read("S2Hand", chip).astype("float32")
        s2, cover[i] = paste(clear, read("S2Hand", src).astype("float32"), stencil)
        xc[i, 2:] = standardise(s2, stats["s2"])
        if real:
            p_real[i] = cloud_prob(clear, f"{chip}")
        p_cloud[i] = cloud_prob(s2, f"{chip}+{src}")
    return chips, x, xc, y, cover, p_real, p_cloud


def blank(x, prob, cut):
    xm = x.copy()
    for i in range(len(x)):
        xm[i, 2:, mask_from(prob[i], cut)] = 0
    return xm


def score(pred, y, where):
    return pooled_iou(pred, np.where(where, y, -1))[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = band_stats("s1s2")
    sources = cloud_sources()
    print(f"{len(sources)} training chips with real clouds to paste", flush=True)
    drop, drop_cut = load_model("s1s2_drop50", 15)

    # 1. choose the mask's probability cut on validation
    _, _, xc, y, cover, _, p_cloud = build("valid", stats, sources, real=False)
    rows = []
    for cut in GRID:
        pred = (predict(drop, blank(xc, p_cloud, cut)) > drop_cut).astype(int)
        masked = np.stack([mask_from(p, cut) for p in p_cloud])
        rows.append({"cut": cut, "flood_iou": round(score(pred, y, y >= 0), 4),
                     "clouds_caught": round(float(masked[cover > 0.5].mean()), 3),
                     "clear_masked": round(float(masked[cover < 0.05].mean()), 3)})
        print(rows[-1], flush=True)
    tuning = pd.DataFrame(rows)
    tuning.to_csv(os.path.join(OUT, "mask_tuning.csv"), index=False)
    cut = float(tuning.cut[tuning.flood_iou.idxmax()])
    print(f"mask cut chosen on validation: {cut}", flush=True)
    del xc, y, cover, p_cloud

    # 2. test: pasted clouds, then the real (unchanged) test chips
    chips, x, xc, y, cover, p_real, p_cloud = build("test", stats, sources)
    pak = np.array([c.startswith("Pakistan") for c in chips])[:, None, None]
    under, outside = (cover > 0.5) & (y >= 0), (cover < 0.05) & (y >= 0)
    radar, radar_cut = load_model("s1_unet", 2)
    optical, optical_cut = load_model("s2_unet", 13)
    step3, step3_cut = load_model("s1s2_unet", 15)
    x1 = np.ascontiguousarray(load_split("test", band_stats("s1"), "s1")[1])
    runs = [("Radar only", lambda: predict(radar, x1) > radar_cut),
            ("Optical only", lambda: predict(optical, np.ascontiguousarray(xc[:, 2:])) > optical_cut),
            ("Radar + optical (Step 3)", lambda: predict(step3, xc) > step3_cut),
            ("Radar + optical (Step 3) + mask", lambda: predict(step3, blank(xc, p_cloud, cut)) > step3_cut),
            ("Sensor dropout", lambda: predict(drop, xc) > drop_cut),
            ("Sensor dropout + mask", lambda: predict(drop, blank(xc, p_cloud, cut)) > drop_cut)]
    rows, preds = [], {}
    for name, run in runs:
        pred = preds[name] = run().astype(int)
        rows.append({"test": "pasted clouds", "model": name, "under_cloud": score(pred, y, under),
                     "outside_cloud": score(pred, y, outside), "all": score(pred, y, y >= 0),
                     "pakistan_all": score(pred, y, (y >= 0) & pak)})
        print(rows[-1], flush=True)
    for name, xin in (("Sensor dropout", x), ("Sensor dropout + mask", blank(x, p_real, cut))):
        pred = (predict(drop, xin) > drop_cut).astype(int)
        sl = chips.index("Sri-Lanka_1049830")
        rows.append({"test": "real test chips", "model": name, "all": score(pred, y, y >= 0),
                     "pakistan_all": score(pred, y, (y >= 0) & pak),
                     "sri_lanka_1049830": pooled_iou(pred[sl], y[sl])[0]})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows).round(3)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)

    masked = np.stack([mask_from(p, cut) for p in p_cloud])
    real = np.stack([mask_from(p, cut) for p in p_real])
    quality = pd.DataFrame([
        {"what": "pasted cloud pixels caught (test)", "share": masked[cover > 0.5].mean()},
        {"what": "clear pixels masked, pasted test", "share": masked[cover < 0.05].mean()},
        {"what": "labelled pixels masked, real test", "share": real[y >= 0].mean()},
        {"what": "labelled pixels masked, real Pakistan test chips", "share": real[(y >= 0) & pak].mean()}]).round(3)
    quality.to_csv(os.path.join(OUT, "mask_quality.csv"), index=False)
    print(res.to_string(index=False))
    print(quality.to_string(index=False))

    show = [i for i in np.argsort(-(under.mean((1, 2)))) if (y[i] == 1).mean() > 0.05][:3]
    show += [i for i in np.flatnonzero(pak[:, 0, 0]) if under[i].mean() > 0.1][:1]
    names = ["hand label", "Radar + optical (Step 3)", "Sensor dropout", "Sensor dropout + mask"]
    fig, axes = plt.subplots(len(show), 6, figsize=(15, 2.7 * len(show)))
    for r, i in enumerate(show):
        rgb = (xc[i, [5, 4, 3]].astype("float32") * np.array(stats["s2"]["std"])[[3, 2, 1], None, None]
               + np.array(stats["s2"]["mean"])[[3, 2, 1], None, None])
        axes[r, 0].imshow((rgb.transpose(1, 2, 0) / 3000).clip(0, 1))
        axes[r, 0].set_title(f"{chips[i]}: optical with pasted cloud", fontsize=8)
        axes[r, 1].imshow(masked[i], cmap="gray")
        axes[r, 1].contour(cover[i].astype("float32"), [0.5], colors="red", linewidths=0.7)
        axes[r, 1].set_title("cloud mask (red: pasted cloud)", fontsize=8)
        for c, name in enumerate(names):
            img = y[i] if c == 0 else preds[name][i]
            axes[r, c + 2].imshow(np.where(y[i] < 0, np.nan, img), cmap="Blues", vmin=0, vmax=1)
            axes[r, c + 2].contour(cover[i].astype("float32"), [0.5], colors="red", linewidths=0.7)
            title = name if c == 0 else f"{name}, IoU under cloud {score(preds[name][i], y[i], under[i]):.2f}"
            axes[r, c + 2].set_title(title, fontsize=7)
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cloud_test.png"), dpi=90)
    print(f"Saved results.csv, mask_tuning.csv, mask_quality.csv, cloud_test.png in {OUT}")


if __name__ == "__main__":
    main()
