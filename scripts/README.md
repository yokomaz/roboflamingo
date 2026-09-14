# Scripts

## Model entry points

- `test_openflamingo.py`: tests the original OpenFlamingo language-generation API. It loads the base OpenFlamingo model and calls `model.generate()`; it prints generated text.
- `infer_roboflamingo.py`: runs the assembled RoboFlamingo policy. It loads CLIP, MPT, the OpenFlamingo backbone, and the policy head; it prints a 7D robot action.
- `inspect_calvin_dataloader.py`: reads one real CALVIN batch and prints tensor shapes.
- `train_roboflamigo.py`: trains the RoboFlamingo policy head and selected Flamingo modules on CALVIN.

For policy inference:

```bash
python scripts/infer_roboflamingo.py --local-files-only
```

For the original language-generation smoke test, use `test_openflamingo.py` only when its model paths are available.

## CALVIN imports

The cloned repository contains two installable Python packages:

```bash
python -m pip install -e calvin/calvin_env --no-deps
python -m pip install -e calvin/calvin_models --no-deps
```

After installation, use the package names defined by CALVIN:

```python
from calvin_agent.datasets.disk_dataset import DiskDataset
from calvin_env.envs.play_table_env import get_env
```

`from calvin import ...` is not the normal CALVIN API. The top-level `calvin/` directory is the repository root, not the package that exports the dataset or environment classes.

