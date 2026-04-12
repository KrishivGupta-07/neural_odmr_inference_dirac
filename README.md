## Setup

```bash
pip install -r requirements.txt
```

## Data

Place data files in the `data/` directory:

- `train.npz`, `val.npz`, `test.npz`, `freq_axis.npy`
- `stress_snr_sweep.npz`, `stress_temp_drift.npz`, `stress_strain.npz`

## Trained models

- Best model checkpoint: `checkpoints/resnet_smooth_seed42_best.pt`
- Compressed models: `checkpoints/v2_student_best.pt`, `checkpoints/v4_pruned70_best.pt`
- Distillation ablation: `checkpoints/v5b_scratch_student.pt`

## Reproducing results

```bash
bash reproduce.sh
```

## Key results

- **Part 1:** ResNet1D base_ch=32, MAE=561.3 uT, R^2=0.8551
- **Part 2:** Best compression V2 Distilled Student, S=322, MAE=571.4 uT
- **Part 3:** SNR cliff at SNR=48 (FP32) and SNR=59 (student) for B=1 mT
