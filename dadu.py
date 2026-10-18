"""
The 2022 floods around Dadu, Johi and Khairpur Nathan Shah (Sindh), mapped by the trained models and
compared with the threshold map of crop_project on the same radar pass.

Date: 23 September 2022, the radar pass of crop_project's orbit (descending, relative orbit 78) with
a clear Sentinel-2 image the same day, like the training chips. Area: crop_project's 50 x 50 km box
around 67.70 E, 26.90 N, on the training chips' own grid (lon/lat, 8.98e-5 degrees, about 10 m).

    radar    sigma-nought rebuilt from the raw GRD file (s1_grd.py, checked against training chips)
    optical  L2A converted towards L1C with the fit from the Pakistan chips (s2_l2a.py); B10 missing;
             clouds and cloud shadows from L2A's scene classification (SCL), grown by 2 pixels
    before   the June 2022 passes of the same orbit (crop_project's pre-monsoon reference), rebuilt the
             same way and averaged in linear power, for the before / after radar model
    models   radar only, optical only, radar + optical (Step 3), sensor dropout + cloud mask (Step 3c),
             radar before / after, each with its own cut from validation; the area is mapped in overlapping 512-pixel tiles
             and the tiles are blended with weights that fall off towards their edges

There are no hand labels here, so the maps are compared with each other: the threshold map of
crop_project (Otsu on the peak date + at least 3 dB darker than June, rebuilt for this date from its
cached radar; with and without its "already dark in June" class, which mixes permanent water with dry
smooth ground), and NDWI > 0 on the clear optical image. Agreement is not accuracy: maps that share an
input (radar, or the optical image) can share a mistake.

    python dadu.py      # writes runs/dadu/ (inputs cached in cache/dadu/)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage

import s1_grd
import s2_l2a
from flood_data import CLIP, band_stats
from unet import UNet

DATE, ORBIT = "2022-09-23", 78
BEFORE = ("2022-06-01", "2022-06-30")                                    # crop_project's June reference
CROP_PROJECT = os.path.join("..", "crop_project")
CROP_CACHE = os.path.join(CROP_PROJECT, "cache", "flood", "vv_67.7_26.9_50000_20_2022-05-01_2022-12-31.npz")
UTM, UTM_BOX = "EPSG:32642", (345900.0, 2951020.0, 395900.0, 3001020.0)    # crop_project's box
RES = 8.983152841195215e-05                                              # the training chips' pixel
OUT, CACHE = os.path.join("runs", "dadu"), os.path.join("cache", "dadu")


def grid():
    w, s, e, n = transform_bounds(UTM, "EPSG:4326", *UTM_BOX)
    shape = (int(round((n - s) / RES)), int(round((e - w) / RES)))
    return from_origin(w, n, RES, RES), shape, "EPSG:4326"


def cached(name, make):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return np.load(path)
    arr = make()
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, arr)
    return arr


def inputs(tr, shape, crs):
    bbox = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(*shape, tr))

    def radar():
        db, shift = s1_grd.radar((bbox, DATE, ORBIT), tr, shape, crs)
        print(f"radar: shifted into place by {shift[0]:.0f} rows, {shift[1]:.0f} columns (median)", flush=True)
        return db.astype("float32")

    def optical():
        raw, scl = s2_l2a.l2a(s2_l2a.find_l2a(bbox, DATE), tr, shape, crs)
        return np.concatenate([raw.clip(0, 65000), scl[None].astype("float32")]).astype("uint16")

    def before():
        days = {}
        for i in s1_grd.catalog().search(collections=["sentinel-1-grd"], bbox=bbox,
                                         datetime=f"{BEFORE[0]}T00:00:00Z/{BEFORE[1]}T23:59:59Z").items():
            if i.properties.get("sat:relative_orbit") == ORBIT:
                days.setdefault(str(i.datetime.date()), True)
        total, count = np.zeros((2,) + tuple(shape), "float32"), np.zeros((2,) + tuple(shape), "float32")
        for day in sorted(days)[-3:]:
            db, shift = s1_grd.radar((bbox, day, ORBIT), tr, shape, crs)
            ok = np.isfinite(db)
            total[ok] += 10 ** (db[ok] / 10)
            count[ok] += 1
            print(f"before: {day}, shifted by {shift[0]:.0f} rows, {shift[1]:.0f} columns", flush=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(count > 0, 10 * np.log10(total / np.maximum(count, 1)), np.nan).astype("float32")

    s1 = cached(f"radar_{DATE}.npy", radar)
    s2 = cached(f"optical_{DATE}.npy", optical)
    pre = cached(f"radar_before_{BEFORE[0]}_{BEFORE[1]}.npy", before)
    return s1, s2[:13], s2[13].astype("uint8"), pre


def standardise(s1, s2, fit, stats, cloud):
    """(15, H, W) float16 model input; radar clipped like the training chips; optical converted."""
    h, w = s1.shape[1:]
    x = np.zeros((15, h, w), dtype="float16")
    for b in range(2):
        v = np.clip(s1[b], *CLIP)
        x[b] = np.nan_to_num((v - stats["s1"]["mean"][b]) / stats["s1"]["std"][b], nan=0.0)
    have = (np.delete(s2, 10, 0) > 0).all(0) & ~cloud
    for k in range(13):
        if k == 10:
            continue                                                   # B10 missing in L2A
        band = s2_l2a.S2_BANDS[k]
        a, c = fit[band]
        v = a * s2[k].astype("float32") + c
        x[k + 2] = np.where(have, (v - stats["s2"]["mean"][k]) / stats["s2"]["std"][k], 0)
    return x


def predict_large(model, x, tile=512, stride=384, batch=4):
    """Water probability for a big area: overlapping tiles blended with edge-tapered weights."""
    c, h, w = x.shape
    ph = tile + int(np.ceil(max(h - tile, 0) / stride)) * stride
    pw = tile + int(np.ceil(max(w - tile, 0) / stride)) * stride
    xp = np.zeros((c, ph, pw), dtype=x.dtype)
    xp[:, :h, :w] = x
    ramp = np.minimum(np.arange(tile) + 1, tile - np.arange(tile)) / (tile - stride)
    wt = np.outer(np.clip(ramp, 0.02, 1), np.clip(ramp, 0.02, 1)).astype("float32")
    acc, norm = np.zeros((ph, pw), "float32"), np.zeros((ph, pw), "float32")
    spots = [(y, x0) for y in range(0, ph - tile + 1, stride) for x0 in range(0, pw - tile + 1, stride)]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(spots), batch):
            part = spots[i:i + batch]
            xb = torch.from_numpy(np.stack([xp[:, y:y + tile, x0:x0 + tile] for y, x0 in part])).float()
            p = torch.softmax(model(xb), 1)[:, 1].numpy()
            for (y, x0), pi in zip(part, p):
                acc[y:y + tile, x0:x0 + tile] += pi * wt
                norm[y:y + tile, x0:x0 + tile] += wt
    return (acc / norm)[:h, :w]


def load_model(name, cin):
    model = UNet(cin=cin)
    model.load_state_dict(torch.load(os.path.join("runs", name, "best.pt"), weights_only=True))
    res = pd.read_csv(os.path.join("runs", name, "results.csv"))
    cut = float(res[(res.set == "valid") & res.method.str.endswith("(tuned)")].method.iloc[0].split()[2])
    return model, cut


def threshold_map():
    """crop_project's rule on DATE: 1 flood (water and >= 3 dB darker than June), 2 water already dark
    in June, 0 not water; its 20 m UTM grid."""
    d = np.load(CROP_CACHE, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(d["vv"])
    db[~np.isfinite(db)] = np.nan

    def smooth(img):
        filled = np.where(np.isfinite(img), img, np.nanmedian(img))
        out = ndimage.median_filter(filled, size=3)
        out[~np.isfinite(img)] = np.nan
        return out

    db = np.stack([smooth(v) for v in db])
    keep = np.isfinite(db).mean(axis=(1, 2)) > 0.9
    db, dates = db[keep], dates[keep]
    ref = (dates >= "2022-06-01") & (dates <= "2022-06-30")
    reference = np.nanmedian(db[ref], axis=0)
    peak = int(np.argmin([np.nanmedian(v) for v in db]))
    v = db[peak][np.isfinite(db[peak])]
    hist, edges = np.histogram(v, bins=256, range=(np.percentile(v, 0.5), np.percentile(v, 99.5)))
    cen = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * cen) / np.maximum(w0, 1)
    m1 = (np.sum(hist * cen) - np.cumsum(hist * cen)) / np.maximum(w1, 1)
    thr = float(cen[np.argmax(w0 * w1 * (m0 - m1) ** 2)])
    i = list(dates.strftime("%Y-%m-%d")).index(DATE)
    water = db[i] < thr
    flood = water & (db[i] < reference - 3)
    out = np.where(flood, 1, np.where(water & (reference < thr), 2, np.where(water, 3, 0))).astype("uint8")
    print(f"threshold map: Otsu {thr:.1f} dB on {dates[peak].date()}, applied to {DATE}", flush=True)
    return out


def to_grid(arr, tr, shape, crs):
    """crop_project's 20 m UTM map onto our grid (nearest); 255 outside its box."""
    dst = np.full(shape, 255, dtype="uint8")
    reproject(arr, dst, src_transform=from_origin(UTM_BOX[0], UTM_BOX[3], 20, 20), src_crs=UTM,
              dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
    return dst


def agree(a, b, where):
    a, b = a & where, b & where
    return (a & b).sum() / max((a | b).sum(), 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, shape, crs = grid()
    print(f"grid {shape[1]} x {shape[0]} pixels", flush=True)
    s1, s2, scl, pre = inputs(tr, shape, crs)
    stats = band_stats("s1s2")
    fit = json.load(open(os.path.join(OUT, "s2_fit.json")))
    cloud = ndimage.binary_dilation(np.isin(scl, s2_l2a.CLOUD_SCL), iterations=2)
    x = standardise(s1, s2, fit, stats, np.zeros(shape, bool))
    xm = standardise(s1, s2, fit, stats, cloud)
    xp = np.zeros((4,) + tuple(shape), dtype="float16")                # radar now, then radar before
    xp[:2] = x[:2]
    for b in range(2):
        v = np.clip(pre[b], *CLIP)
        xp[b + 2] = np.nan_to_num((v - stats["s1"]["mean"][b]) / stats["s1"]["std"][b], nan=0.0)

    maps, probs = {}, {}
    for name, run, cin, xin in (("Radar U-Net", "s1_unet", 2, lambda: np.ascontiguousarray(x[:2])),
                                ("Optical U-Net", "s2_unet", 13, lambda: np.ascontiguousarray(x[2:])),
                                ("Radar + optical U-Net", "s1s2_unet", 15, lambda: x),
                                ("Sensor dropout + cloud mask", "s1s2_drop50", 15, lambda: xm),
                                ("Radar before/after U-Net", "s1pre_unet", 4, lambda: xp)):
        model, cut = load_model(run, cin)
        probs[name] = predict_large(model, xin())
        maps[name] = probs[name] > cut
        print(f"{name}: {maps[name].mean():.1%} of the area mapped as water", flush=True)

    thr = to_grid(threshold_map(), tr, shape, crs)
    inside = thr != 255
    radar_ok = np.isfinite(s1).all(0)
    opt_ok = (np.delete(s2, 10, 0) > 0).all(0)
    clear = np.isin(scl, (4, 5, 6)) & opt_ok
    g, n = s2[2].astype("float32"), s2[7].astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (g - n) / (g + n)
    ndwi_water = ndwi > 0
    maps = {"Threshold (crop_project)": np.isin(thr, (1, 2, 3)),
            "Threshold, without the June-dark class": np.isin(thr, (1, 3)), **maps}
    for c, name in ((1, "flood"), (2, "dark already in June"), (3, "water, less than 3 dB change")):
        m = (thr == c) & inside & opt_ok & clear
        print(f"threshold class {c} ({name}): NDWI calls {ndwi_water[m].mean():.1%} of it water", flush=True)
    common = inside & radar_ok & opt_ok
    lat = rasterio.transform.xy(tr, np.arange(shape[0]), np.zeros(shape[0]))[1]
    px_km2 = (RES * 111.32) ** 2 * np.cos(np.radians(np.array(lat)))[:, None] * np.ones((1, shape[1]))
    rows = []
    for name, m in maps.items():
        rows.append({"map": name, "water_km2": round(float(px_km2[m & common].sum()), 1),
                     "agreement_with_ndwi": round(float(agree(m, ndwi_water, common & clear)), 3),
                     "agreement_with_threshold": round(float(agree(m, maps["Threshold (crop_project)"], common)), 3)})
    rows.append({"map": "NDWI > 0 (clear pixels)", "water_km2": round(float(px_km2[ndwi_water & common & clear].sum()), 1),
                 "agreement_with_ndwi": 1.0,
                 "agreement_with_threshold": round(float(agree(ndwi_water, maps["Threshold (crop_project)"], common & clear)), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "dadu_comparison.csv"), index=False)
    print(res.to_string(index=False))
    print(f"clear optical: {clear[common].mean():.1%} of the area; cloud mask: {cloud[common].mean():.1%}")

    best = maps["Sensor dropout + cloud mask"]
    tif = np.where(common, best.astype("uint8"), 255).astype("uint8")
    with rasterio.open(os.path.join(OUT, f"dadu_water_{DATE}.tif"), "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="uint8", crs=crs, transform=tr, nodata=255, compress="deflate") as dst:
        dst.write(tif, 1)
        dst.write_colormap(1, {0: (240, 236, 226, 255), 1: (31, 106, 163, 255)})

    k = 4                                                              # display at 1/4 resolution
    sub = lambda a: a[::k, ::k]
    rgb = np.stack([s2[3], s2[2], s2[1]], -1).astype("float32") / 2500
    show = lambda m: np.where(sub(common), sub(m).astype(float), np.nan)
    the_thr = maps["Threshold (crop_project)"]
    diff = np.where(best & the_thr, 1, np.where(best & ~the_thr, 2, np.where(~best & the_thr, 3, 0)))
    panels = [("optical, true colour (L2A)", sub(rgb).clip(0, 1), None),
              ("radar VV, sigma-nought (dB)", sub(np.clip(s1[0], -25, 0)), "gray"),
              ("NDWI > 0 (optical water)", show(ndwi_water & clear), "Blues"),
              ("Threshold map (crop_project rule)", show(the_thr), "Blues"),
              ("Radar U-Net", show(maps["Radar U-Net"]), "Blues"),
              ("Sensor dropout + cloud mask U-Net", show(best), "Blues"),
              ("Radar before/after U-Net", show(maps["Radar before/after U-Net"]), "Blues"),
              ("dropout U-Net vs threshold", np.where(sub(common), sub(diff), np.nan),
               ListedColormap(["#efe9dc", "#1f6aa3", "#e08a2c", "#7b3fa0"]))]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            ax.imshow(img)
        else:
            lo, hi = (-25, 0) if "dB" in title else (0, 3) if "vs" in title else (0, 1)
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    axes[1, 3].text(0.01, -0.06, "blue: both water   orange: U-Net only   purple: threshold only",
                    transform=axes[1, 3].transAxes, fontsize=9)
    fig.suptitle(f"Dadu, Johi and Khairpur Nathan Shah, {DATE} (50 x 50 km)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dadu_maps.png"), dpi=80)
    print(f"Saved dadu_comparison.csv, dadu_maps.png, dadu_water_{DATE}.tif in {OUT}")


if __name__ == "__main__":
    main()
