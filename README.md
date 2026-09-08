# Robust Keypoint Recovery for Continuous Sign Language Recognition

This repository contains the implementation and experiments for an MSc project investigating the robustness of keypoint-based Continuous Sign Language Recognition (CSLR) under missing keypoint conditions.

The project is built upon the MSKA-SLR framework and studies how incomplete skeletal keypoint observations affect recognition performance. Two keypoint recovery approaches are evaluated:

- Linear interpolation
- Temporal Denoising Autoencoder (Temporal DAE)

Experiments are conducted on the PHOENIX-2014T dataset.

---

## Project Motivation

Keypoint-based sign language recognition relies on accurate and complete pose information. However, keypoint detection systems may produce missing or unreliable observations due to occlusion, motion blur, or estimation errors.

This project investigates:

1. How different missing-keypoint patterns affect MSKA recognition performance.
2. Whether temporal recovery methods can restore missing information.
3. The effectiveness of learned temporal reconstruction compared with simple interpolation.

---

## Contributions

The main contributions implemented in this project are:

### 1. Missing Keypoint Simulation

Three missingness patterns are introduced:

#### Random Missing

Implemented in:

```
missing/random_mask.py
```

Randomly removes selected keypoints to simulate independent detection failures.

#### Spatial Occlusion

Implemented in:

```
missing/spatial_mask.py
```

Simulates local spatial occlusion by removing keypoints inside a rectangular region in normalized coordinate space.

#### Motion-Aware Missing

Implemented in:

```
missing/motion_mask.py
```

Uses temporal joint displacement to assign higher missing probabilities to fast-moving joints, simulating motion-related detection failures.

---

## Keypoint Recovery Methods

### Linear Interpolation

Implemented in:

```
recovery/interpolation.py
```

Missing keypoints are reconstructed using temporal interpolation between available observations.

Observed keypoints remain unchanged.

---

### Temporal Denoising Autoencoder

Implemented in:

```
models/temporal_dae.py
train_temporal_dae.py
```

The Temporal DAE learns to reconstruct missing keypoints from surrounding temporal information.

The model:

```
Input keypoints + missing mask
        |
        v
Bidirectional GRU
        |
        v
Reconstructed keypoints
```

The model predicts all selected keypoints, but only artificially missing positions are replaced. Original observed keypoints are preserved.

---

## Evaluation Pipeline

The evaluation process is:

```
PHOENIX-2014T keypoints
          |
          v
Generate artificial missing keypoints
          |
          v
Recovery method
(None / Interpolation / Temporal DAE)
          |
          v
MSKA recognition model
          |
          v
WER evaluation
```

The original MSKA input format is preserved. Recovered keypoints are inserted back before being processed by the recognition model.

---

## Experimental Results

### Random Missing Sensitivity

Random missing keypoints cause severe degradation in MSKA performance.

Example:

| Missing ratio | Ensemble WER |
|---|---:|
| Clean input | 19.59 |
| 1.27% missing | 63.99 |
| 2.53% missing | 86.79 |

This demonstrates that MSKA is sensitive to incomplete keypoint observations.

---

## Recovery Comparison

| Missing Type | No Recovery | Interpolation | Temporal DAE |
|---|---:|---:|---:|
| Random (light) | 63.9979 | 19.8025 | **19.5089** |
| Spatial (medium) | 49.5863 | 24.0726 | **21.9909** |
| Motion (light) | 60.0747 | 19.5356 | **19.4556** |

The recovery methods substantially reduce the degradation caused by missing keypoints.

Temporal DAE provides the strongest improvement under structured spatial occlusion, where temporal learning is more beneficial than simple interpolation.

---

## Temporal DAE Training

The Temporal DAE was trained using six corruption conditions:

- Random light
- Random medium
- Spatial light
- Spatial medium
- Motion light
- Motion medium

Final training result:

| Metric | Value |
|---|---:|
| Best epoch | 10 |
| Training masked MSE | 0.00496 |
| Mean validation MSE | 0.00404 |

---

## Repository Structure

```
.
├── missing/
│   ├── random_mask.py
│   ├── spatial_mask.py
│   └── motion_mask.py
│
├── recovery/
│   └── interpolation.py
│
├── models/
│   └── temporal_dae.py
│
├── train_temporal_dae.py
├── evaluate_robustness.py
│
└── results/
    ├── random_missing_sensitivity.csv
    ├── spatial_mask_calibration.csv
    ├── motion_mask_calibration.csv
    ├── temporal_dae_training_summary.csv
    └── recovery_comparison.csv
```

---

## Usage

Example robustness evaluation:

```bash
python3 evaluate_robustness.py \
    --config configs/phoenix-2014t_s2g.yaml \
    --missing_type random \
    --recovery temporal_dae
```

Large files are not included:

- PHOENIX-2014T dataset
- Original MSKA checkpoint
- Temporal DAE checkpoint

These files should be obtained separately.

---

## Acknowledgement

This project is based on:

Guan et al.,
"Multi-stream Keypoint Attention for Sign Language Recognition",
Pattern Recognition, 165, 111602, 2025.

The original MSKA architecture and pretrained model are from the original authors.

The missing-keypoint simulation, recovery methods, Temporal DAE, and robustness evaluation pipeline were implemented as part of this MSc project.
