"""
Sentinel-1 radar for any area, processed the way the Sen1Floods11 chips were: sigma-nought in dB.

The training chips come from Google Earth Engine's Sentinel-1 GRD collection: thermal noise removed,
calibrated to sigma-nought, put on a map grid. Planetary Computer offers the same raw GRD files (with
their calibration and noise tables) over plain HTTPS, so this rebuilds that processing:

    sigma0 = (DN^2 - noise) / A_sigma^2           (per pixel, tables interpolated from the annotation)

then places it on the target grid with the scene's ground control points (bilinear, in linear power)
and converts to dB. The control points sit tens of metres off (they assume a ground height that is not
the real one, and radar places a pixel by its distance from the satellite), so the result is then
shifted into place: block by block it is matched against Planetary Computer's terrain-corrected product
of the same pass (gamma-nought, RTC, well positioned but about 1 dB brighter and with the noise left
in), and the measured shifts are smoothed and applied. Brightness stays our own sigma-nought.
There is no terrain correction otherwise: fine on flat ground such as the Indus plain.

    python s1_grd.py --check      # compare with the training chips of events whose exact pass exists
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import requests
from rasterio.control import GroundControlPoint
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from scipy import ndimage

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


@lru_cache(maxsize=1)
def catalog():
    return pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)


def find_scenes(bbox, date, orbit=None, collection="sentinel-1-grd"):
    """Scenes of one UTC day over bbox (lon/lat), optionally only one relative orbit."""
    items = catalog().search(collections=[collection], bbox=bbox,
                             datetime=f"{date}T00:00:00Z/{date}T23:59:59Z").items()
    return [i for i in items if orbit is None or i.properties.get("sat:relative_orbit") == orbit]


@lru_cache(maxsize=16)
def _xml(href):
    return ET.fromstring(requests.get(href, timeout=180).content)


def _floats(node, tag):
    return np.array(node.find(tag).text.split(), dtype="float64")


def _lut(vectors, rows, cols):
    """Tables given as (line, pixels, values) vectors -> values at every (row, col) of the window."""
    lines = np.array([v[0] for v in vectors], dtype="float64")
    per_line = np.stack([np.interp(cols, v[1], v[2]) for v in vectors])
    i = np.clip(np.searchsorted(lines, rows) - 1, 0, len(lines) - 2)
    w = np.clip((rows - lines[i]) / (lines[i + 1] - lines[i]), 0, 1)[:, None]
    return per_line[i] * (1 - w) + per_line[i + 1] * w


def _calibration(item, pol, rows, cols):
    root = _xml(item.assets[f"schema-calibration-{pol}"].href)
    vec = [(int(v.find("line").text), _floats(v, "pixel"), _floats(v, "sigmaNought"))
           for v in root.iter("calibrationVector")]
    return _lut(vec, rows, cols)


def _noise(item, pol, rows, cols):
    """Thermal noise: one range table (older files), or range x azimuth tables (files from 2018 on)."""
    root = _xml(item.assets[f"schema-noise-{pol}"].href)
    old = [(int(v.find("line").text), _floats(v, "pixel"), _floats(v, "noiseLut")) for v in root.iter("noiseVector")]
    if old:
        return _lut(old, rows, cols)
    rng = [(int(v.find("line").text), _floats(v, "pixel"), _floats(v, "noiseRangeLut"))
           for v in root.iter("noiseRangeVector")]
    noise = _lut(rng, rows, cols)
    az = np.ones_like(noise)
    for v in root.iter("noiseAzimuthVector"):
        l0, l1 = int(v.find("firstAzimuthLine").text), int(v.find("lastAzimuthLine").text)
        c0, c1 = int(v.find("firstRangeSample").text), int(v.find("lastRangeSample").text)
        r = (rows >= l0) & (rows <= l1)
        c = (cols >= c0) & (cols <= c1)
        if r.any() and c.any():
            line, lut = _floats(v, "line"), _floats(v, "noiseAzimuthLut")
            vals = np.interp(rows[r], line, lut) if len(line) > 1 else np.full(r.sum(), lut[0])
            az[np.ix_(r, c)] = vals[:, None]
    return noise * az


def _window(gcps, bbox, width, height, margin=64):
    """Radar rows/cols covering bbox, from a quadratic fit of the ground control points."""
    lon = np.array([g.x for g in gcps]); lat = np.array([g.y for g in gcps])
    A = np.stack([np.ones_like(lon), lon, lat, lon * lat, lon ** 2, lat ** 2], 1)
    cr = np.linalg.lstsq(A, np.array([g.row for g in gcps]), rcond=None)[0]
    cc = np.linalg.lstsq(A, np.array([g.col for g in gcps]), rcond=None)[0]
    x0, y0, x1, y1 = bbox
    pts = [(x, y) for x in np.linspace(x0, x1, 5) for y in np.linspace(y0, y1, 5)]
    P = np.array([[1, x, y, x * y, x * x, y * y] for x, y in pts])
    r, c = P @ cr, P @ cc
    r0, r1 = int(max(r.min() - margin, 0)), int(min(r.max() + margin, height))
    c0, c1 = int(max(c.min() - margin, 0)), int(min(c.max() + margin, width))
    return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None


def sigma0(items, dst_transform, dst_shape, dst_crs):
    """(2, H, W) sigma-nought in dB (VV, VH) on the target grid, NaN where no scene covers it."""
    out = np.full((2,) + tuple(dst_shape), np.nan, dtype="float32")
    bbox = transform_bounds(dst_crs, "EPSG:4326", *rasterio.transform.array_bounds(*dst_shape, dst_transform))
    for item in items:
        for b, pol in enumerate(("vv", "vh")):
            with rasterio.open(item.assets[pol].href) as src:
                gcps, gcrs = src.gcps
                win = _window(gcps, bbox, src.width, src.height)
                if win is None:
                    continue
                r0, r1, c0, c1 = win
                dn = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype("float64")
            rows, cols = np.arange(r0, r1, dtype="float64"), np.arange(c0, c1, dtype="float64")
            lin = (dn ** 2 - _noise(item, pol, rows, cols)) / _calibration(item, pol, rows, cols) ** 2
            lin = np.maximum(lin, 1e-5).astype("float32")            # below the noise floor -> -50 dB
            lin[dn == 0] = np.nan                                    # no data at the scene's edges
            local = [GroundControlPoint(row=g.row - r0, col=g.col - c0, x=g.x, y=g.y) for g in gcps]
            dst = np.full(dst_shape, np.nan, dtype="float32")
            reproject(lin, dst, gcps=local, src_crs=gcrs, dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan, SRC_METHOD="GCP_TPS")
            fill = np.isnan(out[b]) & np.isfinite(dst)
            out[b][fill] = 10 * np.log10(dst[fill])
    return out


def gamma0(items, dst_transform, dst_shape, dst_crs):
    """(2, H, W) Planetary Computer RTC gamma-nought in dB (VV, VH) on the target grid."""
    out = np.full((2,) + tuple(dst_shape), np.nan, dtype="float32")
    bounds = rasterio.transform.array_bounds(*dst_shape, dst_transform)
    for item in items:
        for b, pol in enumerate(("vv", "vh")):
            with rasterio.open(item.assets[pol].href) as src:
                x0, y0, x1, y1 = transform_bounds(dst_crs, src.crs, *bounds)
                win = rasterio.windows.from_bounds(x0 - 200, y0 - 200, x1 + 200, y1 + 200, src.transform)
                win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
                if win.width < 2 or win.height < 2:
                    continue
                arr = src.read(1, window=win)
                dst = np.full(dst_shape, np.nan, dtype="float32")
                reproject(arr, dst, src_transform=src.window_transform(win), src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs=dst_crs,
                          resampling=Resampling.bilinear, src_nodata=0, dst_nodata=np.nan)
            fill = np.isnan(out[b]) & np.isfinite(dst) & (dst > 0)
            out[b][fill] = 10 * np.log10(dst[fill])
    return out


def _shift(a, b, search):
    """Integer (dy, dx) that best moves b onto a (normalised cross-correlation via FFT)."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    c = np.fft.fftshift(c)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    w = c[cy - search:cy + search + 1, cx - search:cx + search + 1]
    dy, dx = np.unravel_index(np.argmax(w), w.shape)
    return dy - search, dx - search, w.max() / a.size


def align(ours, ref, block=256, search=16):
    """Shift ours (2, H, W, dB) onto ref's positions with a smooth field of block-wise shifts."""
    h, w = ours.shape[1:]
    a = ndimage.uniform_filter(np.nan_to_num(ref[0], nan=np.nanmedian(ref[0])), 3)
    b = ndimage.uniform_filter(np.nan_to_num(ours[0], nan=np.nanmedian(ours[0])), 3)
    ys, xs = np.arange(0, max(h - block, 0) + 1, block // 2), np.arange(0, max(w - block, 0) + 1, block // 2)
    dy, dx, good = (np.zeros((len(ys), len(xs))) for _ in range(3))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ra, rb = a[y:y + block, x:x + block], b[y:y + block, x:x + block]
            ok = np.isfinite(ref[0][y:y + block, x:x + block]).mean() > 0.9 and np.isfinite(ours[0][y:y + block, x:x + block]).mean() > 0.9
            if ok:
                dy[i, j], dx[i, j], good[i, j] = _shift(ra, rb, search)
    keep = good > 0.3
    if not keep.any():
        return ours, (np.nan, np.nan)
    my, mx = np.median(dy[keep]), np.median(dx[keep])
    dy[~keep], dx[~keep] = my, mx
    dy, dx = ndimage.median_filter(dy, 3, mode="nearest"), ndimage.median_filter(dx, 3, mode="nearest")
    gy, gx = (ys + block / 2) , (xs + block / 2)
    # the smooth shift field at every pixel (float32 to keep a 50 km area in memory)
    fy = np.broadcast_to(np.interp(np.arange(h), gy, np.arange(len(gy))).astype("float32")[:, None], (h, w))
    fx = np.broadcast_to(np.interp(np.arange(w), gx, np.arange(len(gx))).astype("float32")[None, :], (h, w))
    sy = ndimage.map_coordinates(dy.astype("float32"), [fy, fx], order=1, mode="nearest", output=np.float32)
    sx = ndimage.map_coordinates(dx.astype("float32"), [fy, fx], order=1, mode="nearest", output=np.float32)
    del fy, fx
    src_y = np.arange(h, dtype="float32")[:, None] - sy
    src_x = np.arange(w, dtype="float32")[None, :] - sx
    del sy, sx
    out = np.empty_like(ours)
    for k in range(len(ours)):
        lin = (10 ** (ours[k] / 10)).astype("float32")
        moved = ndimage.map_coordinates(np.nan_to_num(lin, nan=0), [src_y, src_x], order=1, mode="constant",
                                        cval=0, output=np.float32)
        valid = ndimage.map_coordinates(np.isfinite(lin).astype("float32"), [src_y, src_x], order=1,
                                        mode="constant", output=np.float32) > 0.99
        out[k] = np.where(valid, 10 * np.log10(np.maximum(moved, 1e-5)), np.nan)
    return out, (float(my), float(mx))


def radar(bbox_date_orbit, dst_transform, dst_shape, dst_crs):
    """Sigma-nought in dB (VV, VH), positioned with the RTC product of the same pass."""
    bbox, date, orbit = bbox_date_orbit
    ours = sigma0(find_scenes(bbox, date, orbit), dst_transform, dst_shape, dst_crs)
    rtc = gamma0(find_scenes(bbox, date, orbit, "sentinel-1-rtc"), dst_transform, dst_shape, dst_crs)
    return align(ours, rtc)


def check(events, out_csv):
    """Our sigma-nought and Planetary Computer's gamma-nought against the training chips, pixel by pixel."""
    meta = {f["properties"]["location"]: f["properties"]
            for f in json.load(open(os.path.join("data", "Sen1Floods11_Metadata.geojson")))["features"]}
    rows = []
    for ev in events:
        p = meta[ev]
        chips = sorted(f.replace("_S1Hand.tif", "") for f in os.listdir(os.path.join("data", "S1Hand"))
                       if f.startswith(ev + "_"))
        date, orbit = p["s1_date"].replace("/", "-"), int(p["rel_orbit_num"])
        diffs = {"sigma0": [[], []], "gamma0": [[], []]}
        ref_all, ours_all = [[], []], [[], []]
        for chip in chips:
            path = os.path.join("data", "S1Hand", f"{chip}_S1Hand.tif")
            with rasterio.open(path) as s:
                ref, tr, crs, shape = s.read().astype("float32"), s.transform, s.crs, (s.height, s.width)
            bbox = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(*shape, tr))
            grd = find_scenes(bbox, date, orbit)
            if not grd:
                continue
            rtc = gamma0(find_scenes(bbox, date, orbit, "sentinel-1-rtc"), tr, shape, crs)
            ours, _ = align(sigma0(grd, tr, shape, crs), rtc)
            for b in range(2):
                ok = np.isfinite(ref[b]) & np.isfinite(ours[b]) & np.isfinite(rtc[b]) & (ref[b] > -40) & (ours[b] > -40)
                diffs["sigma0"][b].append((ours[b] - ref[b])[ok])
                diffs["gamma0"][b].append((rtc[b] - ref[b])[ok])
                sm = lambda x: ndimage.uniform_filter(np.nan_to_num(x, nan=-15), 5)   # speckle averaged out
                ref_all[b].append(sm(ref[b])[ok][::5]); ours_all[b].append(sm(ours[b])[ok][::5])
        for b, pol in enumerate(("VV", "VH")):
            if not diffs["sigma0"][b]:
                continue
            r, o = np.concatenate(ref_all[b]), np.concatenate(ours_all[b])
            for prod in ("sigma0", "gamma0"):
                d = np.concatenate(diffs[prod][b])
                rows.append({"event": ev, "band": pol, "product": prod, "chips": len(diffs[prod][b]),
                             "median_diff_db": round(float(np.median(d)), 2),
                             "mean_abs_diff_db": round(float(np.mean(np.abs(d))), 2),
                             "corr_5x5": round(float(np.corrcoef(r, o)[0, 1]), 3) if prod == "sigma0" else None})
                print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--events", nargs="+", default=["Bolivia", "Somalia", "Nigeria"])
    a = ap.parse_args()
    if a.check:
        check(a.events, os.path.join("runs", "dadu", "s1_check.csv"))
