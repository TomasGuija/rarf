# RARF for BraTS inpainting

This folder contains the 3D medical-image pipeline used to apply [Region-Aware Rectified Flows (RARF)](../README.md) to the BraTS inpainting challenge. It provides the BraTS dataset and data module, training configurations, single-case inference, healthy-mask generation, and checkpoint evaluation.

## Method overview

![Overview of the RARF BraTS pipeline](../doc/RARF_Overview.svg)

RARF learns a rectified flow for filling a missing 3D region while using the visible T1-weighted anatomy as context. The BraTS adapter centers and crops each NIfTI volume, normalizes it using a foreground intensity percentile, and supplies the target, missing-region mask, and task-specific loss mask to the generic RARF training module. See the repository-level [README](../README.md#configurable-rarf-modes) for the flow-path and conditioning modes.

## Expected data layout

Each case is a directory of co-registered NIfTI volumes. Training discovers either the unnumbered mask pair or one or more numbered mask variants:

```text
/path/to/training/
└── <case-id>/
    ├── <case-id>-t1n.nii.gz
    ├── <case-id>-mask-unhealthy.nii.gz     # used to generate healthy variants
    ├── <case-id>-mask-healthy.nii.gz       # unnumbered variant (optional)
    ├── <case-id>-mask.nii.gz               # unhealthy + healthy mask
    ├── <case-id>-mask-healthy-0000.nii.gz  # numbered variant (optional)
    └── <case-id>-mask-0000.nii.gz
```

For training, `mask-healthy[-NNNN]` is the region scored by the challenge
objective, while its paired `mask[-NNNN]` is the complete region removed from
the input. When numbered pairs exist, training samples one at random on each
access and validation consistently uses the first. `mask-unhealthy` is needed
by the mask-generation tool but is not carried through the training pipeline.

Inference data uses the challenge-style voided image and its complete inpainting mask:

```text
/path/to/inference/
└── <case-id>/
    ├── <case-id>-t1n-voided.nii.gz
    └── <case-id>-mask.nii.gz
```

All volumes belonging to a case must already have the same orientation, shape, and voxel grid. The loader can alternatively use `train_csv` and `val_csv`; both CSVs must contain `case_id`, `t1n`, `variant_id`, `mask`, and `mask_healthy` columns. Relative paths are resolved against the corresponding dataset root.

## Training

`brats/train.py` uses Lightning CLI, so values in a YAML file can be overridden on the command line.

```bash
python brats/train.py fit \
  --config configs/brats/default.yaml \
  --data.data_dir /path/to/training \
  --data.train_split 0.95
```

Without a separate validation root, the data module makes a reproducible train/validation split using `data.train_split`. To use a dedicated validation set:

```bash
python brats/train.py fit \
  --config configs/brats/default.yaml \
  --data.data_dir /path/to/training \
  --data.val_data_dir /path/to/validation \
  --data.train_split 1.0
```

To resume a run, pass a Lightning checkpoint:

```bash
python brats/train.py fit \
  --config configs/brats/default.yaml \
  --data.data_dir /path/to/training \
  --data.train_split 0.95 \
  --ckpt_path /path/to/last.ckpt
```

For multi-GPU training, also override the device count and strategy as needed, for example `--trainer.devices 2 --trainer.strategy ddp`. Checkpoints and CSV logs are written beneath `checkpoints/` and `outputs/` according to the selected configuration.

BraTS preprocessing uses the same percentile bounds as the challenge metric:

```yaml
data:
  robust_percentile_lower: 0.5
  robust_percentile: 99.5
```

The bounds are computed from all voxels of the full voided volume before
cropping. Inputs and targets are clipped to those bounds and mapped to `[0,1]`.
Inference reads the settings from the checkpoint, merges samples in normalized
space, and converts the final prediction back to raw intensities.

Reconstruction supervision accepts either one loss and weight or equally sized
lists. Supported reconstruction losses are `mae`, `mse`, and `ssim`.

## Provided tools

### Generate healthy mask variants

[`generate_healthy_masks.py`](tools/generate_healthy_masks.py) derives transformed templates from unhealthy-mask components and places them in healthy brain tissue. It writes paired numbered `mask-healthy-NNNN` and combined `mask-NNNN` volumes:

```bash
python brats/tools/generate_healthy_masks.py /path/to/training \
  --samples-per-case 5 \
  --output-root /path/to/training-with-variants
```

### Run inference on one case

[`infer.py`](tools/infer.py) restores preprocessing settings from the checkpoint, inpaints one case, and saves `<case-id>-rarf-inpainted.nii.gz`:

```bash
python brats/tools/infer.py \
  --checkpoint /path/to/model.ckpt \
  --dataset-root /path/to/inference \
  --case-id <case-id> \
  --output-dir outputs/inference \
  --sample-steps 32 \
  --sampling-strategy mean \
  --num-samples 30 \
  --seed 0
```

Sampling strategies are:

- `normal`: generate one sample; requires `--num-samples 1`.
- `mean`: return the voxel-wise mean of all generated samples.
- `closest_to_mean`: return the generated sample closest to their voxel-wise mean inside the mask.

Samples are generated sequentially to limit GPU memory. When `--seed` is set,
sample seeds increase from that value. `--case-id` expects the complete case
directory name, for example:

```bash
--case-id PSEUDO_VAL_BraTS-GLI-00114-000
```

It is not a numeric dataset index. To select by position instead, omit
`--case-id` and pass a zero-based index such as `--index 0`; positions follow
the sorted list of complete cases found under `--dataset-root`. The default
device is CUDA when available, otherwise CPU.

## Docker submission

The submission image uses [`brats/docker/inference.py`](docker/inference.py), a
small inference entry point. The image contains only PyTorch, NiBabel, the RARF model
and flow implementation, the inference entry point, and one checkpoint.

The container expects challenge-style case directories at
`/input`. It writes exactly one prediction per case into the flat `/output`
directory:

```text
/input/<case-id>/<case-id>-t1n-voided.nii.gz
/input/<case-id>/<case-id>-mask.nii.gz
/output/<case-id>-t1n-inference.nii.gz
```

### Build

The checkpoint is the only required external build artifact. It may use any
name or directory inside the Docker build context; pass its context-relative
path with `MODEL_CHECKPOINT`:

```bash
docker build \
  -f brats/docker/Dockerfile \
  --build-arg MODEL_CHECKPOINT=checkpoints/final.ckpt \
  -t brats-rarf-submission \
  .
```

The resulting image embeds the checkpoint and therefore needs only the input
and output mounts at runtime.

### Run

Single-sample inference is the default:

```bash
LOCAL_INPUT=/path/to/challenge-input \
LOCAL_OUTPUT=/path/to/empty-output \
bash brats/scripts/run_docker_submission.sh
```

Multiple deterministic samples can either be averaged or reduced to the
generated sample closest to their voxel-wise mean inside the inpainting mask:

```bash
LOCAL_INPUT=/path/to/challenge-input \
LOCAL_OUTPUT=/path/to/empty-output \
NUM_SAMPLES=4 \
MERGE_STRATEGY=closest_to_mean \
bash brats/scripts/run_docker_submission.sh
```

The runtime interface is:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `SAMPLE_STEPS` | `4` | Midpoint integration points. Must be at least two. |
| `NUM_SAMPLES` | `1` | Number of deterministic samples generated sequentially. |
| `MERGE_STRATEGY` | `mean` | Either `mean` or `closest_to_mean`. |
| `SEED` | `0` | Base seed; each case and sample receives a stable derived seed. |

EMA weights are selected automatically when the checkpoint declares EMA use
and contains `ema_state_dict`; otherwise raw model weights are used. Crop shape,
normalization percentile, model architecture, flow path, and time conditioning
are restored directly from the checkpoint.

### Evaluate a checkpoint

[`evaluate.py`](evaluation/evaluate.py) evaluates generated healthy tissue with MSE, MAE, SSIM, and PSNR. It writes per-case results to `metrics.csv` and aggregate means and standard deviations to `metrics_summary.json`:

```bash
python brats/evaluation/evaluate.py \
  --checkpoint /path/to/model.ckpt \
  --data-root /path/to/validation \
  --output-dir outputs/brats-evaluation/model \
  --sample-steps 32
```

Add `--save-outputs` to retain generated and voided NIfTI volumes for visual inspection. If `--data-root` is omitted, the tool uses `val_data_dir` stored in the checkpoint.


## Challenge results

Below are the main challenge results obtained through the validation data predictions submission. Each value represents the mean over 219 validation samples. The values were obtained by sampling 50 independent predictions and obtaining the voxel-wise mean prediction. For further discussions and study over these results, please refer to the challenge paper.

| Submission / split | MSE ↓ | MAE ↓ | SSIM ↑ | PSNR ↑ |
| --- | ---: | ---: | ---: | ---: |
| BraTS challenge validation set | 0.006 | 0.018 | 0.832 | 24.008 |

## References

1. Ujjwal Baid, Satyam Ghodasara, Suyash Mohan, et al. **“The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification.”** *arXiv preprint arXiv:2107.02314*, 2021. [arXiv:2107.02314](https://arxiv.org/abs/2107.02314).

2. Florian Kofler, Felix Meissen, Felix Steinbauer, et al. **“The Brain Tumor Segmentation (BraTS) Challenge: Local Synthesis of Healthy Brain Tissue via Inpainting.”** *arXiv preprint arXiv:2305.08992*, 2024. [arXiv:2305.08992](https://arxiv.org/abs/2305.08992).
