#!/usr/bin/env python
"""Evaluate a trained RoboFlamingo policy with CALVIN rollouts."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CALVIN_ROOT = ROOT / "calvin"
sys.path[:0] = [
    str(ROOT),
    str(CALVIN_ROOT / "calvin_models"),
    str(CALVIN_ROOT / "calvin_env"),
]

from roboflamingo import create_model


DEFAULT_VISION_ENCODER = ROOT / (
    "models/language_conditioned/clip_vit_l14/"
    "modelscope_clip_vit_large_patch14"
)
DEFAULT_LANGUAGE_MODEL = ROOT / "models/language_conditioned/mpt_1b_dolly"
DEFAULT_BACKBONE_CHECKPOINT = ROOT / (
    "models/language_conditioned/"
    "openflamingo_3b_vitl_mpt1b_langinstruct/checkpoint.pt"
)
DEFAULT_DATASET = ROOT / "calvin/dataset/calvin_debug_dataset"


class RoboFlamingoPolicy:
    def __init__(self, model, image_processor, tokenizer, device, dtype):
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.hidden = None
        self.goal = None
        self.tokens = None

    def reset(self):
        self.hidden = None
        self.goal = None
        self.tokens = None

    def step(self, obs, goal):
        if goal != self.goal:
            self.goal = goal
            self.tokens = self.tokenizer(
                [f"<image>{goal.strip()}<|endofchunk|>"],
                max_length=32,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        pixels = self.image_processor(
            images=obs["rgb_obs"]["rgb_static"],
            return_tensors="pt",
        )["pixel_values"]
        vision_x = pixels.unsqueeze(1).to(self.device, dtype=self.dtype)
        input_ids = self.tokens["input_ids"].to(self.device)
        attention_mask = self.tokens["attention_mask"].to(self.device).bool()

        with torch.inference_mode():
            prediction, self.hidden = self.model(
                vision_x,
                input_ids,
                attention_mask,
                hidden=self.hidden,
            )

        action = prediction[0, -1].float().cpu().numpy()
        action[6] = 1.0 if action[6] > 0.0 else -1.0
        return action


def load_policy(args, device):
    for path in (
        args.vision_encoder,
        args.language_model,
        args.backbone_checkpoint,
        args.checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    model, image_processor, tokenizer = create_model(
        vision_encoder_path=str(args.vision_encoder),
        lang_encoder_path=str(args.language_model),
        tokenizer_path=str(args.language_model),
        checkpoint_path=str(args.backbone_checkpoint),
        use_local_files=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    return RoboFlamingoPolicy(model, image_processor, tokenizer, device, dtype)


def rollout(env, policy, task_oracle, task, instruction, max_steps):
    policy.reset()
    obs = env.get_obs()
    start_info = env.get_info()
    for _ in range(max_steps):
        obs, _, _, current_info = env.step(policy.step(obs, instruction))
        solved = task_oracle.get_task_info_for_set(
            start_info,
            current_info,
            {task},
        )
        if solved:
            return True
    return False


def evaluate(args, policy):
    try:
        import hydra
        from omegaconf import OmegaConf
        from tqdm.auto import tqdm

        from calvin_agent.evaluation.multistep_sequences import get_sequences
        from calvin_agent.evaluation.utils import (
            count_success,
            get_env_state_for_initial_condition,
        )
        from calvin_env.envs.play_table_env import get_env
    except ImportError as error:
        raise RuntimeError(
            "CALVIN evaluation dependencies are missing. Install the local "
            "calvin_env and calvin_models packages before running evaluation."
        ) from error

    validation_dir = args.dataset / "validation"
    if not (validation_dir / ".hydra" / "merged_config.yaml").is_file():
        raise FileNotFoundError(
            f"CALVIN environment config not found under {validation_dir}"
        )

    conf_dir = CALVIN_ROOT / "calvin_models/conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )
    env = get_env(validation_dir, show_gui=False)
    sequences = get_sequences(args.num_sequences, num_workers=args.sequence_workers)

    results = []
    task_success = Counter()
    task_attempts = Counter()
    try:
        for initial_state, tasks in tqdm(sequences, desc="CALVIN evaluation"):
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            completed = 0
            for task in tasks:
                task_attempts[task] += 1
                instruction = annotations[task][0]
                if not rollout(
                    env,
                    policy,
                    task_oracle,
                    task,
                    instruction,
                    args.max_steps,
                ):
                    break
                task_success[task] += 1
                completed += 1
            results.append(completed)
    finally:
        env.close()

    chain_success = count_success(results)
    total_success = sum(task_success.values())
    total_attempts = sum(task_attempts.values())
    metrics = {
        "num_sequences": len(results),
        "average_successful_sequence_length": float(np.mean(results)),
        "subtask_success_rate": total_success / max(total_attempts, 1),
        "chain_success_rate": {
            str(index): rate for index, rate in enumerate(chain_success, start=1)
        },
        "task_success": {
            task: {
                "success": task_success[task],
                "attempts": attempts,
                "rate": task_success[task] / attempts,
            }
            for task, attempts in sorted(task_attempts.items())
        },
    }
    return metrics


def print_metrics(metrics):
    print(
        "average successful sequence length: "
        f"{metrics['average_successful_sequence_length']:.3f}"
    )
    print(f"subtask success rate: {metrics['subtask_success_rate'] * 100:.1f}%")
    print("success rates for consecutive instructions:")
    for length, rate in metrics["chain_success_rate"].items():
        print(f"  {length}/5: {rate * 100:.1f}%")
    print("per-task success rates:")
    for task, result in metrics["task_success"].items():
        print(
            f"  {task}: {result['success']}/{result['attempts']} "
            f"({result['rate'] * 100:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Measure a trained RoboFlamingo policy's CALVIN success rate."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--vision-encoder",
        type=Path,
        default=DEFAULT_VISION_ENCODER,
    )
    parser.add_argument(
        "--language-model",
        type=Path,
        default=DEFAULT_LANGUAGE_MODEL,
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=DEFAULT_BACKBONE_CHECKPOINT,
    )
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--sequence-workers", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/roboflamingo_evaluation/results.json"),
    )
    args = parser.parse_args()

    if args.num_sequences < 1:
        parser.error("--num-sequences must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.precision == "bf16":
        args.precision = "fp32"
    print(f"device: {device}, precision: {args.precision}")
    policy = load_policy(args, device)
    metrics = evaluate(args, policy)
    print_metrics(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"saved results to {args.output}")


if __name__ == "__main__":
    main()
