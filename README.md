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

## Step 1: a U-Net on Sentinel-1

```powershell
python train.py        # about an hour on a laptop CPU; checkpoints every epoch, re-run to resume
python evaluate.py     # scores on test, Bolivia and the Pakistan test chips, next to Otsu
```

A four-level U-Net (1.94 million weights) on radar only (VV, VH in dB, clipped to [-35, 5] and
standardised with training statistics). Each epoch takes one random 256 x 256 crop per training chip;
cross-entropy with water weighted 9.5x (water is 9.5% of training pixels), no-data pixels ignored.
The best epoch by validation IoU is kept (epoch 26 of 36, early stopping after 10 epochs without
improvement). Every chip is then mapped whole (512 x 512).

| Pooled IoU of water | Otsu | U-Net |
|---|---|---|
| Validation (89 chips) | 0.51 | **0.62** |
| Test (90 chips) | 0.55 | **0.62** |
| Bolivia, an unseen event (15 chips) | **0.72** | 0.68 |
| Pakistan test chips (6) | 0.18 | **0.22** |

- **On the familiar events the U-Net wins:** it misses 44% less water than Otsu on the test chips
  (0.35 million missed pixels against 0.63 million), for about 12% more false alarms.
- **On Bolivia, a flood it never saw, the simple threshold wins.** A model trained on ten events
  does not automatically transfer to an eleventh: the domain-shift lesson.
- **In Pakistan both fail.** The U-Net cuts Otsu's false alarms by about a third (0.24 million
  pixels against 0.35 million), but sand dunes and dry, smooth ground are as dark to radar as water.
  Optical imagery sees them as bright, which is the reason for the radar + optical step.
- Validation IoU jumped between 0.37 and 0.62 from epoch to epoch, a sign that 252 chips are few;
  augmentation is the next step.

![Pakistan test chips](runs/s1_unet/pakistan_test.png)

## Roadmap

- [x] Download the hand-labelled chips; score the Otsu baseline
- [x] U-Net on Sentinel-1 (VV, VH) vs Otsu: test, Bolivia, Pakistan
- [ ] Better training: augmentation, weighted / Dice loss for the rare water class
- [ ] Radar + optical (Sentinel-2) vs radar alone
- [ ] A pretrained encoder vs training from scratch
- [ ] Apply the model to the 2022 Dadu floods and compare with the threshold map from crop_project
