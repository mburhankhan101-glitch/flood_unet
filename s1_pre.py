"""
A "before" radar image for every hand-labelled chip, so a model can compare the flood day with
the weeks before it.

For each flood event the passes on the SAME relative orbit as the flood image (same viewing geometry,
so no fake change) between 90 and 12 days before it are found on Planetary Computer; the latest three
are rebuilt as sigma-nought exactly like the radar for Dadu (s1_grd.py: raw GRD, noise removed,
calibrated, shifted into place against the RTC of the same pass) on each chip's own grid, and averaged
in linear power, which also averages out part of the speckle. Chips with no such pass get no file and
are treated as "before unknown".

    python s1_pre.py        # writes data/S1PreHand/<chip>_S1PreHand.tif (VV, VH dB); re-run to resume
"""
import json
import os
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds

import s1_grd

OUT = os.path.join("data", "S1PreHand")
LOG = os.path.join("runs", "prepost", "pre_dates.csv")
EARLIEST, LATEST, MAX_PASSES = 90, 12, 3            # days before the flood image; passes averaged


def passes(bbox, flood_day, orbit):
    """Dates (latest first) with both a GRD and an RTC scene on this orbit in the window."""
    start, end = flood_day - timedelta(days=EARLIEST), flood_day - timedelta(days=LATEST)
    found = defaultdict(lambda: {"grd": [], "rtc": []})
    for coll, key in (("sentinel-1-grd", "grd"), ("sentinel-1-rtc", "rtc")):
        items = s1_grd.catalog().search(collections=[coll], bbox=bbox,
                                        datetime=f"{start}T00:00:00Z/{end}T23:59:59Z").items()
        for i in items:
            if i.properties.get("sat:relative_orbit") == orbit:
                found[i.datetime.date()][key].append(i)
    days = sorted((d for d, v in found.items() if v["grd"]), reverse=True)
    return [(d, found[d]["grd"], found[d]["rtc"]) for d in days[:MAX_PASSES]]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    meta = [f["properties"] for f in json.load(open(os.path.join("data", "Sen1Floods11_Metadata.geojson")))["features"]]
    log = []
    for p in meta:
        ev = {"Cambodia": "Mekong"}.get(p["location"], p["location"]).replace(" ", "-")   # Mekong chips
        chips = sorted(f.replace("_S1Hand.tif", "") for f in os.listdir(os.path.join("data", "S1Hand"))
                       if f.startswith(ev + "_"))
        if not chips:
            continue
        grids = {}
        for chip in chips:
            with rasterio.open(os.path.join("data", "S1Hand", f"{chip}_S1Hand.tif")) as s:
                grids[chip] = (s.transform, (s.height, s.width), s.crs)
        b = np.array([transform_bounds(g[2], "EPSG:4326", *rasterio.transform.array_bounds(*g[1], g[0]))
                      for g in grids.values()])
        bbox = [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]
        flood_day = date.fromisoformat(p["s1_date"].replace("/", "-"))
        orbit = int(p["rel_orbit_num"])
        found = passes(bbox, flood_day, orbit)
        print(f"{ev}: flood image {flood_day}, orbit {orbit}; before: {[str(d) for d, _, _ in found] or 'none'}", flush=True)
        done = 0
        for chip in chips:
            path = os.path.join(OUT, f"{chip}_S1PreHand.tif")
            if os.path.exists(path):
                done += 1
                continue
            tr, shape, crs = grids[chip]
            total, count, used = np.zeros((2,) + shape), np.zeros((2,) + shape), 0
            for day, grd, rtc in found:
                for attempt in range(3):           # download links expire; re-sign and retry a failed read
                    try:
                        g = [s1_grd.fresh(i) for i in grd]
                        r = [s1_grd.fresh(i) for i in rtc]
                        db = s1_grd.sigma0(g, tr, shape, crs)
                        if np.isfinite(db).mean() >= 0.05 and r:
                            db, _ = s1_grd.align(db, s1_grd.gamma0(r, tr, shape, crs))
                        break
                    except (rasterio.errors.RasterioIOError, requests.RequestException) as e:
                        print(f"  {chip} {day}: read failed ({str(e)[:60]}), retry {attempt + 1}", flush=True)
                else:
                    continue                                  # this pass could not be read
                if np.isfinite(db).mean() < 0.05:
                    continue                                  # this pass does not reach the chip
                ok = np.isfinite(db)
                total[ok] += 10 ** (db[ok] / 10)
                count[ok] += 1
                used += 1
            if used == 0:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                pre = np.where(count > 0, 10 * np.log10(total / np.maximum(count, 1)), np.nan).astype("float32")
            with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1], count=2, dtype="float32",
                               crs=crs, transform=tr, nodata=np.nan, compress="deflate", predictor=3) as dst:
                dst.write(pre)
            done += 1
        log.append({"event": ev, "flood_image": flood_day, "orbit": orbit, "chips": len(chips), "with_before": done,
                    "before_dates": " ".join(str(d) for d, _, _ in found)})
        print(f"  {done} of {len(chips)} chips have a before image", flush=True)
        pd.DataFrame(log).to_csv(LOG, index=False)


if __name__ == "__main__":
    main()
