# Sparse E2VID

Event-based video reconstruction using sparse convolutions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the pre-trained checkpoint and place it in `checkpoints/`:
```
checkpoints/last_0150.pth
```

3. Prepare your dataset `.h5` files and place them in the `data/` folder.  
   *(The `.h5` files are not tracked by Git due to size limits, but should be present locally in `data/` to run the scripts.)*

## Datasets

### Test data
The test datasets (e.g. `bike_bay_hdr.h5`, `boxes.h5`, etc.) should be placed in:
```
data/
├── bike_bay_hdr.h5
├── boxes.h5
├── desk.h5
└── ...
```

### Training data
The training dataset is **not included** in this repository due to its large size (~65 GB).

You can download it from Baidu Netdisk:

- **Link:** https://pan.baidu.com/s/1l4K0ukKyiR_ZBc6bctZgSQ
- **Extraction code:** `mtvk`
- **File:** `tr_data_full.tar.xz`

After downloading, extract the archive and place the `.h5` files as follows:

```bash
tar -xf tr_data_full.tar.xz
mkdir -p data/train data/noise
mv tr_data_full/*.h5 data/train/
# Place noise calibration .h5 files in data/noise/
```

Expected structure:
```
data/
├── train/          # Training sequences (.h5 files)
│   ├── 000000000_out.h5
│   ├── 000000001_out.h5
│   └── ...
├── noise/          # Noise calibration data (.h5 files)
│   ├── noise1.h5
│   └── ...
└── bike_bay_hdr.h5   # Test files (kept in data/ root for convenience)
```

## Usage

### Training

To train the model from scratch:

```bash
python train.py --tr_path ./data/train --noise_path ./data/noise --device cuda:0
```

Checkpoints will be saved to `./checkpoints/training/` (best and last).

Key training arguments:
| Argument       | Default | Description                          |
|----------------|---------|--------------------------------------|
| `--tr_path`    | ./data/train | Path to training .h5 sequences |
| `--noise_path` | ./data/noise | Path to noise calibration data |
| `--save_path`  | ./checkpoints/training | Where to save checkpoints |
| `--epochs`     | 50      | Number of training epochs            |
| `--lr`         | 1e-3    | Learning rate                        |
| `--bs`         | 1       | Batch size                           |
| `--embed_dim`  | 16      | Embedding dimension                  |

### Single dataset evaluation with metrics

```bash
python test.py --data_path ./data/bike_bay_hdr.h5 --device cuda:0
```

### Run inference on all test datasets

```bash
python test_all_datasets.py --device cuda:0
```

This will evaluate the model on all available test sequences, print a summary table with TC, LPIPS, SSIM and MSE metrics, and save the results to `./test_results.csv`.

### Fast inference (no metrics)

```bash
python inference.py --data_path ./data/bike_bay_hdr.h5 --output_dir ./output --device cuda:0
```

This runs optimized inference without computing any metrics and saves the reconstructed frames to `--output_dir`.

### Optimized inference (FP16 / torch.compile)

```bash
python inference_optim.py --fp16 --compile --device cuda:0
```

## Repository Structure

```
sparse_e2vid/
├── checkpoints/          # Pre-trained model weights
│   └── last_0150.pth
├── data/                 # Input .h5 event datasets (local only)
│   ├── train/            # Training sequences
│   ├── noise/            # Noise calibration data
│   └── *.h5              # Test datasets
├── models/               # Model definitions
│   └── small_e2v3_SubMsparse5.py
├── tools/                # Utilities (metrics, timers, losses)
│   ├── metrics.py
│   ├── timers.py
│   ├── loss.py
│   └── loss_fn.py
├── dataset.py            # Training dataset loader
├── train.py              # Training script
├── test.py               # Single dataset evaluation
├── test_all_datasets.py  # Batch evaluation on all test sets
├── inference.py          # Standard inference script
├── inference_optim.py    # Optimized inference (FP16 / compile)
├── requirements.txt
└── README.md
```

## Model Parameters

| Argument       | Default | Description                          |
|----------------|---------|--------------------------------------|
| `--embed_dim`  | 16      | Embedding dimension                  |
| `--num_bins`   | 5       | Number of bins for voxel grid        |
| `--in_chans`   | 5       | Input channels                       |
| `--out_chans`  | 2       | Output channels                      |
| `--kernel_size`| 3       | Convolution kernel size              |
| `--ev_rate`    | 0.1     | Event rate for voxel generation      |
| `--device`     | cuda:0  | Device to use for inference          |

## Citation

If you use this code, please cite the original work.
