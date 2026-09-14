# RoboFlamingo

A local RoboFlamingo reproduction workspace for CALVIN language-conditioned robot imitation.

## Repository layout

- `roboflamingo/modeling.py`: assembled CLIP, Perceiver, OpenFlamingo, MPT, and policy head.
- `roboflamingo/data.py`: CALVIN sliding-window Dataset and DataLoader.
- `scripts/infer_roboflamingo.py`: 7D robot-action inference.
- `scripts/test_openflamingo.py`: original OpenFlamingo text-generation smoke test.
- `scripts/inspect_calvin_dataloader.py`: inspect a real CALVIN batch.
- `scripts/train_roboflamigo.py`: CALVIN policy training entry point.

## Setup

Use a server-provided PyTorch/CUDA image when available. Conda channels only
apply to Conda packages; model and OpenFlamingo packages are installed by pip.

```bash
conda env create -f environment.yml
conda activate roboflamingo
```

The tested local environment is `aarch64` with NVIDIA GB10 and CUDA 13.0.
For the same platform, install the tested PyTorch set first:

```bash
pip install -r requirements-cu130-aarch64.txt
```

On another server, install the PyTorch build matching its preconfigured image, then run:

```bash
pip install -r requirements.txt
```

Check compatibility before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Install CALVIN from its upstream repository or from a local clone:

```git
git clone https://github.com/mees/calvin.git
pip install -e calvin/calvin_env --no-deps
pip install -e calvin/calvin_models --no-deps
```

Large model weights and CALVIN datasets are intentionally excluded from Git. Pass their server paths to scripts explicitly.

## Local verification

```bash
python scripts/inspect_calvin_dataloader.py --batch-size 2 --window-size 32
python scripts/infer_roboflamingo.py --local-files-only
```

## Training

```bash
python scripts/train_roboflamigo.py \
  --dataset /path/to/calvin_debug_dataset \
  --batch-size 32 \
  --window-size 32 \
  --steps 1000
```

See `scripts/README.md` for the distinction between inference scripts and CALVIN imports.

