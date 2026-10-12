"""
Download the hand-labelled part of Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020) from its
public Google Cloud bucket, over plain HTTPS: no Google account or gsutil needed.

    data/S1Hand/            Sentinel-1 VV, VH in dB, 512 x 512 at 10 m (446 chips)
    data/S2Hand/            Sentinel-2, 13 bands, TOA reflectance x 10000
    data/LabelHand/         hand labels: 1 water, 0 not water, -1 no data
    data/S1OtsuLabelHand/   the dataset authors' Otsu-threshold water map (a ready-made baseline)
    data/JRCWaterHand/      permanent water from the JRC Global Surface Water layer
    data/splits/            train / valid / test / bolivia chip lists

About 1.75 GB. Files already downloaded with the right size are skipped, so it can be re-run.

    python download.py
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "sen1floods11"
BASE = "v1.1/data/flood_events/HandLabeled"
FOLDERS = ["S1Hand", "S2Hand", "LabelHand", "S1OtsuLabelHand", "JRCWaterHand"]
SPLITS = ["train", "valid", "test", "bolivia"]
OUT = "data"


def listing(prefix):
    """All (name, size) under a prefix, following the API's pages."""
    items, token = [], ""
    while True:
        url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix={prefix}"
               f"&fields=nextPageToken,items(name,size)&maxResults=1000" + (f"&pageToken={token}" if token else ""))
        page = json.load(urllib.request.urlopen(url, timeout=60))
        items += [(i["name"], int(i["size"])) for i in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return items


def fetch(job):
    url, path, size = job
    if os.path.exists(path) and (size < 0 or os.path.getsize(path) == size):   # size -1: small file, size unknown
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(4):
        try:
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            return size
        except OSError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {url}")


def main():
    jobs = []
    for folder in FOLDERS:
        for name, size in listing(f"{BASE}/{folder}/"):
            jobs.append((f"https://storage.googleapis.com/{BUCKET}/{name}",
                         os.path.join(OUT, folder, os.path.basename(name)), size))
    for split in SPLITS:
        name = f"v1.1/splits/flood_handlabeled/flood_{split}_data.csv"
        jobs.append((f"https://storage.googleapis.com/{BUCKET}/{name}", os.path.join(OUT, "splits", f"{split}.csv"), -1))
    total = sum(s for _, _, s in jobs if s > 0)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB in total", flush=True)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, got in enumerate(pool.map(fetch, jobs), 1):
            done += max(got, 0)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} files, {done / 1e6:.0f} MB downloaded ({time.time() - t0:.0f} s)", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
