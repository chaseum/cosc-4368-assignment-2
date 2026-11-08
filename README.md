# Imagewoof CNN

Custom sequential CNNs, trained from scratch at 128 x 128:

| Architecture | Channels per block | Convolutions | Parameters |
| --- | --- | --- | --- |
| `baseline` | 32, 64, 128 | 6 | 288,746 |
| `deep` | 32, 64, 128, 256 | 8 | 1,175,786 |
| `deep_spatial` | 32, 64, 128, 256 | 8 | 1,183,466 |
| `deep_pointwise` | 32, 64, 128, 256 | 9 | 1,241,834 |
| `deep_2233` | 32, 64, 128, 256 | 10 | 1,913,834 |
| `wide_2233` | 64, 128, 256, 512 | 10 | 7,643,082 |
| `wide_22333` | 64, 128, 256, 512, 512 | 13 | 14,724,042 |

Each block has two Conv2d -> BatchNorm2d -> ReLU sequences, then 2x2 max pooling.
The baseline and deep heads use global average pooling, flatten, dropout (0.3), and one linear layer
with 10 output logits. There are no pretrained weights, skip connections, or attention.
`deep_spatial` instead pools to a 2 x 2 grid and uses a 1024-to-10 linear layer.
It uses fixed 4 x 4 average pooling on the 8 x 8 features, equivalent to adaptive
2 x 2 pooling for these fixed inputs, with deterministic CUDA backward support.
`deep_pointwise` adds a 256-to-256 1 x 1 Conv/BatchNorm/ReLU block before global pooling.
`deep_2233` uses 2, 2, 3, and 3 convolutions in its four blocks. Channels, pooling,
BatchNorm/ReLU ordering, and the global-average-pooling classifier match `deep`.
It adds one 3 x 3 convolution in each of the last two blocks, increasing convolution
MACs by approximately 36%. Existing architectures and checkpoints retain their layouts.
`--dropout` sets the head dropout explicitly; it defaults to 0.3 and is saved with new checkpoints.

Both architectures follow the HW2 constraints: standard layers in one sequential path,
randomly initialized weights, 128 x 128 RGB input, and 10 class logits. Global average
pooling is explicitly allowed. The tests check allowed layer types, sequential execution,
spatial dimensions, and output shape. The training script uses only the provided local
training/validation folders and no automated hyperparameter or architecture search.
Start with the baseline and choose between models using held-out validation accuracy;
compliance and passing shape checks do not establish classification quality.

## Windows / NVIDIA setup

Run these PowerShell commands from the repository root. A separate environment avoids
replacing packages in your global Python. Python 3.10 is installed on the development PC.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

This CUDA 12.8 wheel command comes from the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/#v2100).
The installed driver reports CUDA 13.2 support; the wheel need not have the same version
number displayed by `nvidia-smi`. Verify CUDA using the final command before training.

## Checks and training

The long-run recipe uses `--augmentation strong`: bilinear 10-degree rotation, random resized
crop (scale 0.5-1.0), flip, color jitter 0.3, and random erasing (p=0.25). CUDA training uses
mixed precision; validation and `predict` stay float32. Use `--patience` above `--epochs`
because accuracy climbs late in the cosine schedule.

```powershell
.\.venv\Scripts\python.exe train.py --architecture wide_2233 --augmentation strong --normalize --label-smoothing 0.1 --weight-decay 0.01 --dropout 0.3 --epochs 150 --patience 151 --batch-size 64 --seed 42 --output runs/wide_2233_strong
.\.venv\Scripts\python.exe train.py --architecture wide_22333 --augmentation strong --normalize --label-smoothing 0.1 --weight-decay 0.01 --dropout 0.3 --epochs 150 --patience 151 --batch-size 64 --seed 42 --output runs/wide_22333_strong
```

Compare them with `train.py --evaluate <run>/model.pth`. `--train-on-all` then refits the
chosen recipe on train + validation and saves the final epoch; its logged validation accuracy
is not held out, so keep the matching split run as the honest estimate.

`--init-from <checkpoint>` fine-tunes an existing model at a low `--lr` instead of starting from
random weights, reusing that checkpoint's architecture, dropout, and normalization.

To test training-set RGB normalization with the best eight-convolution recipe:

```powershell
.\.venv\Scripts\python.exe train.py --architecture deep --augmentation original --normalize --label-smoothing 0.05 --weight-decay 0.0001 --dropout 0.3 --epochs 60 --batch-size 32 --seed 42 --output runs/deep_smooth_normalized
.\.venv\Scripts\python.exe train.py --evaluate runs/deep_smooth_normalized/model.pth
```

`--normalize` computes pixel-weighted RGB mean and population standard deviation from
the unaugmented training images only, after scaling to [0, 1]. The same `(image - mean) / std`
transformation is applied after training augmentation, to validation tensors, and inside
`predict`. Statistics are saved in `config.json` and the checkpoint. Evaluation restores
them from the checkpoint automatically. Old checkpoints without these statistics retain
their original division-by-255 preprocessing; do not normalize their inputs retroactively.
This is an isolated preprocessing experiment; the architecture and other settings stay
the same. Compare against the 82.01% smoothing-only reference. The launcher below does
not enable normalization automatically.

To test the 2-2-3-3 architecture with the best existing training recipe:

```powershell
.\.venv\Scripts\python.exe train.py --architecture deep_2233 --augmentation original --label-smoothing 0.05 --weight-decay 0.0001 --dropout 0.3 --epochs 60 --batch-size 32 --seed 42 --output runs/deep_2233_smooth
```

This starts from random weights and saves separately. Compare its best validation
accuracy against `runs/deep_smooth_only` (82.01%). More depth is an experiment, not a
guarantee of improvement. The fixed ablation launcher below keeps its existing three trials.

To run the three predefined ablations with one command:

```powershell
.\.venv\Scripts\python.exe run_ablations.py
```

This runs three fixed trials sequentially on CUDA, all with dropout 0.3:

| Run | Architecture | Training augmentation | Label smoothing |
| --- | --- | --- | ---: |
| `deep_crop` | `deep` | Conservative crop + original transforms | 0 |
| `deep_spatial_control` | `deep_spatial` | Original transforms | 0 |
| `deep_crop_smooth` | `deep` | Conservative crop + original transforms | 0.05 |

Each starts from random weights. Compare crop + smoothing against crop alone; compare
the spatial control and crop-only models against the historical deep reference.
The third trial always runs; there is no automatic decision based on the first result.
All use 60 epochs, batch size 32,
seed 42, and the existing optimizer, schedule and early stopping. Budget approximately
80 minutes based on the earlier deep run; actual runtime depends on system load.

A fresh timestamped folder under `runs/` contains a separate checkpoint/config/history
for each trial, snapshots of the model/data/training/launcher code, and
`comparison.csv` / `comparison.md`, including augmentation and smoothing settings.
Comparisons update after each trial, include earlier `runs/baseline` and `runs/deep`
results when available, and keep historical results separate from newly completed runs.
Earlier runs lack some recorded settings, so the comparison labels them unrecorded.
The launcher stops on failure; it preserves completed and partial results and does not
resume interrupted training. It never overwrites an old run or promotes a winning model.
Use the suite's saved `model.py` with the selected checkpoint for packaging.

HW2 prohibits hyperparameter/architecture search tools. This launcher executes only
three predefined manual ablations, with no generated combinations or adaptive tuning.
The PDF does not explicitly address scripted ablation launchers; do not treat this
distinction as instructor confirmation that automation is permitted.

For a short execution check, or a different common training budget:

```powershell
.\.venv\Scripts\python.exe run_ablations.py --smoke-test
.\.venv\Scripts\python.exe run_ablations.py --epochs 60 --batch-size 16 --output runs/ablations_batch16
```

Smoke results use tiny subsets, omit historical comparisons, and cannot establish accuracy.
Do not edit the training/model source while the launcher is running. Future repeated
validation comparisons can overstate hidden-test performance.

The crop uses scale `(0.82, 1.0)` and aspect ratio `(0.90, 1.10)` at 128 x 128.
Color jitter stays at 0.1 so that the crop-only trial isolates cropping. Validation
remains `ToTensor()` with no random transformations. Label smoothing affects training
loss only; all validation losses use ordinary cross-entropy.

Light erasing is available as a separate, optional follow-up, not part of the default
three trials. It adds erasing after tensor conversion with probability 0.12, area
`(0.01, 0.05)`, aspect ratio `(0.7, 1.4)`, and random noise fill:

```powershell
.\.venv\Scripts\python.exe train.py --architecture deep --augmentation crop_erase --epochs 60 --batch-size 32 --seed 42 --output runs/deep_crop_erase
```

MixUp, CutMix, and weight averaging are not implemented. `predict` averages each image with
its horizontal mirror. The model's required inference interface is unchanged.

```powershell
# Architecture, gradients, checkpoint loading, and raw-image inference checks.
.\.venv\Scripts\python.exe -m unittest -v

# One epoch on 32 training and 32 validation images; checks CUDA execution.
.\.venv\Scripts\python.exe train.py --device cuda --smoke-test --output runs/smoke

# Deliberately fit 64 training images with augmentation and dropout disabled.
.\.venv\Scripts\python.exe train.py --device cuda --overfit 64 --epochs 100 --patience 100 --weight-decay 0 --output runs/overfit

# Compare both architectures under the same training settings and provided split.
.\.venv\Scripts\python.exe train.py --device cuda --architecture baseline --epochs 60 --batch-size 32 --seed 42 --output runs/baseline
.\.venv\Scripts\python.exe train.py --device cuda --architecture deep --epochs 60 --batch-size 32 --seed 42 --output runs/deep

# Evaluate each saved best checkpoint through the actual submission interface.
.\.venv\Scripts\python.exe train.py --device cuda --evaluate runs/baseline/model.pth
.\.venv\Scripts\python.exe train.py --device cuda --evaluate runs/deep/model.pth
```

Output folders must be new to prevent accidental replacement of experiments. Choose a
different `--output` when repeating a run. If CUDA runs out of memory, reduce
`--batch-size` to 16 for both comparison runs. Use `--device cpu` for CPU checks.

Training uses AdamW (learning rate 0.001, weight decay 0.0001), a cosine learning-rate
schedule, cross-entropy, and early stopping after 15 epochs without improved validation
accuracy. It preserves the existing data pipeline and augmentation. Seeds and deterministic
PyTorch algorithms are enabled. Exact results across different hardware/software versions
are not guaranteed. CUDA training uses mixed precision; validation and inference use float32.

Each run saves:

- `model.pth`: best checkpoint by validation accuracy, with architecture, seed, epoch,
  class names, and model state. It contains no optimizer state and cannot resume training.
- `config.json`: arguments, class order, and environment versions.
- `history.csv`: epoch losses, accuracies, learning rate, and elapsed time.
- `best_metrics.json`: best epoch, per-class recall/support, and confusion matrix
  (rows are actual classes, columns are predictions, both in `classes.txt` order).

Training accuracy includes augmentation and dropout, while validation uses eval mode
without augmentation. These accuracies do not measure identical conditions. Debug runs
are marked in their configuration and metrics. Overfit mode evaluates the same training
subset; neither debug mode measures full held-out performance.

## Submission interface

`model.py` is self-contained and exposes `load_model(path="model.pth")` and
`predict(model, images)`. Loading defaults to CPU and eval mode; use `.to("cuda")` if
desired. `predict` accepts uint8 RGB arrays shaped `(N, 128, 128, 3)`, converts to NCHW
float32 divided by 255, applies the checkpoint's normalization, averages logits for each image
and its horizontal mirror, and returns an int64 array of class IDs.

After choosing the model, package **only** `model.py` and that run's `model.pth` at the
root of the assignment ZIP. Do not use a smoke/overfit checkpoint. Keep the supplied
validation split for model selection, and do not obtain external or hidden test images.
