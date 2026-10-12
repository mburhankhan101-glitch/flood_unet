# flood_unet

Flood mapping from Sentinel-1 radar (and Sentinel-2) with a U-Net, trained and tested on the
hand-labelled part of [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
(Bonafilia et al., CVPR Workshops 2020), then applied to the 2022 floods around Dadu, Pakistan.
A companion to [crop_project](https://github.com/mburhankhan101-glitch/crop_project), where the
Dadu floods were mapped with a threshold.

```powershell
python download.py     # 446 hand-labelled chips + splits, about 1.75 GB, into data/
python explore.py      # split statistics, the Otsu baseline, example chips
```

## Data
446 chips of 512 x 512 pixels at 10 m from 11 flood events (Bolivia, Ghana, India, Mekong, Nigeria,
**Pakistan**, Paraguay, Somalia, Spain, Sri Lanka, USA): Sentinel-1 VV/VH in dB, Sentinel-2 (13 bands),
and hand labels (water, not water, no data). Splits: 252 train, 89 validation, 90 test, and the
15 Bolivia chips as an event never seen in training. Pakistan has 28 chips (16 / 6 / 6).

## The baseline to beat
The dataset ships an Otsu-threshold water map for every chip. Against the hand labels (IoU of water):

| Split | IoU |
|---|---|
| Train / valid / test | 0.45 / 0.51 / 0.55 |
| Bolivia (unseen event) | 0.70 |
| **Pakistan chips** | **0.15** |

On the Pakistan chips Otsu finds 0.28 million water pixels correctly but marks 1.44 million dry
pixels as water. Otsu always splits a chip in two, even one with almost no water, and dry smooth
ground in the Indus plain is as dark to radar as water.

![VV histogram](runs/vv_histogram.png)
![Pakistan chips](runs/pakistan_examples.png)
