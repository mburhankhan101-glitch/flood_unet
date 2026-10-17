"""
Sentinel-2 for any area from Planetary Computer's L2A product, made as close as possible to the L1C
chips the model was trained on.

The training chips are L1C: reflectance at the top of the atmosphere, 13 bands including B10 (cirrus).
L1C files are not freely downloadable without an account, so this uses L2A: the atmosphere (haze,
mostly in blue) is estimated and removed, and B10 is dropped. Since processing baseline 04.00
(January 2022) L2A values also carry an offset of +1000, removed here.

The gap is measured on the Pakistan chips (28 June 2017, the same day as an L2A scene): a straight
line per band, L1C = a x L2A + b, is fitted on the 16 training chips' clear pixels and checked on the
12 validation and test chips. Then the dropout model maps those 12 labelled chips four ways, with the
same radar each time: the real L1C; L1C without B10; raw L2A; converted L2A (both without B10).

    python s2_l2a.py --check      # writes runs/dadu/s2_fit.csv and s2_check.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

from flood_data import S2_BANDS, band_stats, load_split, pooled_iou, read, split_chips
from s1_grd import catalog
from train import predict
from unet import UNet

CLEAR_SCL = (4, 5, 6, 7)          # vegetation, bare soil, water, unclassified
CLOUD_SCL = (3, 8, 9, 10)         # cloud shadow, cloud (medium, high), thin cirrus


def pc_name(band):
    """Planetary Computer's asset name: B1 -> B01, B8A stays."""
    return band if band == "B8A" else f"B{int(band[1:]):02d}"


def find_l2a(bbox, date):
    items = catalog().search(collections=["sentinel-2-l2a"], bbox=bbox,
                             datetime=f"{date}T00:00:00Z/{date}T23:59:59Z").items()
    return list(items)


def _read(href, dst_transform, dst_shape, dst_crs, resampling):
    with rasterio.open(href) as src:
        x0, y0, x1, y1 = transform_bounds(dst_crs, src.crs, *rasterio.transform.array_bounds(*dst_shape, dst_transform))
        win = rasterio.windows.from_bounds(x0 - 300, y0 - 300, x1 + 300, y1 + 300, src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
        dst = np.zeros(dst_shape, dtype="float32")
        if win.width >= 2 and win.height >= 2:
            reproject(src.read(1, window=win).astype("float32"), dst, src_transform=src.window_transform(win),
                      src_crs=src.crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=resampling,
                      src_nodata=0, dst_nodata=0)
    return dst


def l2a(items, dst_transform, dst_shape, dst_crs):
    """(13, H, W) reflectance x 10000 in the L1C band order with B10 = 0 (missing), and SCL (H, W).
    Nearest-neighbour resampling, as Earth Engine uses by default for the training chips."""
    out = np.zeros((13,) + tuple(dst_shape), dtype="float32")
    scl = np.zeros(dst_shape, dtype="uint8")
    for item in items:
        offset = 1000.0 if item.properties.get("s2:processing_baseline", "00.00") >= "04.00" else 0.0
        empty = (out == 0).all(0)
        if not empty.any():
            break
        for k, band in enumerate(S2_BANDS):
            if band == "B10":
                continue
            v = _read(item.assets[pc_name(band)].href, dst_transform, dst_shape, dst_crs, Resampling.nearest)
            v = np.where(v > 0, np.maximum(v - offset, 1), 0)
            out[k][empty] = v[empty]
        s = _read(item.assets["SCL"].href, dst_transform, dst_shape, dst_crs, Resampling.nearest)
        scl[empty] = s[empty].astype("uint8")
    return out, scl


def convert(s2, fit):
    """Raw L2A (13, H, W) -> L1C-like, band by band; B10 stays 0 (missing)."""
    out = s2.copy()
    for k, band in enumerate(S2_BANDS):
        if band in fit:
            a, b = fit[band]
            out[k] = np.where(s2[k] > 0, np.maximum(a * s2[k] + b, 1), 0)
    return out


def _chip_l2a(chip, date):
    with rasterio.open(os.path.join("data", "S2Hand", f"{chip}_S2Hand.tif")) as s:
        l1c, tr, crs, shape = s.read().astype("float32"), s.transform, s.crs, (s.height, s.width)
    bbox = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(*shape, tr))
    raw, scl = l2a(find_l2a(bbox, date), tr, shape, crs)
    return l1c, raw, scl


def check(out_dir):
    date = "2017-06-28"
    os.makedirs(out_dir, exist_ok=True)
    pk = {s: [c for c in split_chips(s) if c.startswith("Pakistan")] for s in ("train", "valid", "test")}
    data = {c: _chip_l2a(c, date) for s in pk for c in pk[s]}
    print(f"L2A loaded for {len(data)} Pakistan chips", flush=True)

    rows, fit = [], {}
    for k, band in enumerate(S2_BANDS):
        if band == "B10":
            continue
        def pairs(chips):
            xs, ys = [], []
            for c in chips:
                l1c, raw, scl = data[c]
                ok = np.isin(scl, CLEAR_SCL) & (l1c[k] > 0) & (raw[k] > 0)
                xs.append(raw[k][ok][::3]); ys.append(l1c[k][ok][::3])
            return np.concatenate(xs), np.concatenate(ys)
        x, y = pairs(pk["train"])
        a, b = np.polyfit(x, y, 1)
        fit[band] = (float(a), float(b))
        xh, yh = pairs(pk["valid"] + pk["test"])
        rows.append({"band": band, "a": round(a, 3), "b": round(b, 1),
                     "median_l1c_minus_l2a": round(float(np.median(yh - xh)), 1),
                     "median_l1c_minus_converted": round(float(np.median(yh - (a * xh + b))), 1),
                     "corr": round(float(np.corrcoef(xh, yh)[0, 1]), 3)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "s2_fit.csv"), index=False)
    json.dump(fit, open(os.path.join(out_dir, "s2_fit.json"), "w"), indent=1)

    # the dropout model on the 12 labelled validation + test chips, four optical versions
    stats = band_stats("s1s2")
    m, s = (np.array(stats["s2"][k], "float32")[:, None, None] for k in ("mean", "std"))
    model = UNet(cin=15)
    model.load_state_dict(torch.load(os.path.join("runs", "s1s2_drop50", "best.pt"), weights_only=True))
    res = pd.read_csv(os.path.join("runs", "s1s2_drop50", "results.csv"))
    cut = float(res[(res.set == "valid") & res.method.str.endswith("(tuned)")].method.iloc[0].split()[2])
    chips = pk["valid"] + pk["test"]
    xs, ys = [], []
    for split in ("valid", "test"):
        names, x, y = load_split(split, stats, "s1s2")
        keep = [i for i, c in enumerate(names) if c.startswith("Pakistan")]
        xs.append(x[keep]); ys.append(y[keep])
    x, y = np.concatenate(xs), np.concatenate(ys)
    b10 = S2_BANDS.index("B10") + 2

    def std(img):
        z = (img - m) / s
        z[:, ~(np.delete(img, 10, 0) > 0).all(0)] = 0     # no optical data
        z[10] = 0                                         # B10 missing
        return z

    versions = {"L1C (training type)": x,
                "L1C without B10": x.copy(),
                "raw L2A, no B10": x.copy(),
                "converted L2A, no B10": x.copy()}
    versions["L1C without B10"][:, b10] = 0
    for i, c in enumerate(chips):
        _, raw, _ = data[c]
        versions["raw L2A, no B10"][i, 2:] = std(raw)
        versions["converted L2A, no B10"][i, 2:] = std(convert(raw, fit))
    rows = []
    for name, xin in versions.items():
        pred = (predict(model, xin) > cut).astype(int)
        iou, tp, fp, fn = pooled_iou(pred, y)
        rows.append({"optical": name, "pakistan_iou": round(iou, 3), "found": tp, "false_alarms": fp, "missed": fn})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "s2_check.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        check(os.path.join("runs", "dadu"))
