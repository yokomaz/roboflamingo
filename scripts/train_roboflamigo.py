#!/usr/bin/env python
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboflamingo import CalvinCollator, CalvinDataset, create_model


DEFAULT_VISION_ENCODER = Path(
    "models/language_conditioned/clip_vit_l14/"
    "modelscope_clip_vit_large_patch14"
)
DEFAULT_LANGUAGE_MODEL = Path("models/language_conditioned/mpt_1b_dolly")
DEFAULT_BACKBONE_CHECKPOINT = Path(
    "models/language_conditioned/"
    "openflamingo_3b_vitl_mpt1b_langinstruct/checkpoint.pt"
)
HF_VISION_ENCODER = "openai/clip-vit-large-patch14"
HF_LANGUAGE_MODEL = "anas-awadalla/mpt-1b-redpajama-200b-dolly"
HF_BACKBONE_REPO = "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(dataset, image_processor, tokenizer, batch_size, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=CalvinCollator(image_processor, tokenizer),
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def configure_trainable_parameters(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules = [
        model.flamingo.perceiver,
        model.flamingo.lang_encoder.gated_cross_attn_layers,
        model.policy_head,
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found")
    return trainable


def action_loss(prediction, target):
    pose_loss = F.mse_loss(prediction[..., :6], target[..., :6])
    gripper_target = ((target[..., 6] + 1.0) / 2.0).clamp(0.0, 1.0)
    gripper_loss = F.binary_cross_entropy_with_logits(
        prediction[..., 6], gripper_target
    )
    return pose_loss + gripper_loss, pose_loss, gripper_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(3, device=device)
    batches = 0
    for batch in loader:
        prediction, _ = model(
            batch["rgb_static"].to(device, non_blocking=True),
            batch["input_ids"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True),
        )
        loss, pose_loss, gripper_loss = action_loss(
            prediction, batch["actions"].to(device, non_blocking=True)
        )
        totals += torch.stack((loss, pose_loss, gripper_loss))
        batches += 1
    model.train()
    return (totals / max(batches, 1)).cpu().tolist()


def resolve_model_source(source, default_path, remote_id, local_files_only):
    path = Path(source)
    if path.exists():
        return str(path)
    if local_files_only:
        raise FileNotFoundError(f"Local model path not found: {path}")
    return remote_id if path == default_path else source


def resolve_checkpoint(source, local_files_only):
    path = Path(source)
    if path.is_file():
        return str(path)
    if local_files_only or path != DEFAULT_BACKBONE_CHECKPOINT:
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_BACKBONE_REPO, "checkpoint.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("calvin/dataset/calvin_debug_dataset"),
    )
    parser.add_argument(
        "--vision-encoder",
        default=str(DEFAULT_VISION_ENCODER),
    )
    parser.add_argument(
        "--language-model",
        default=str(DEFAULT_LANGUAGE_MODEL),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        default=str(DEFAULT_BACKBONE_CHECKPOINT),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/roboflamingo_debug"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder = resolve_model_source(
        args.vision_encoder,
        DEFAULT_VISION_ENCODER,
        HF_VISION_ENCODER,
        args.local_files_only,
    )
    language_model = resolve_model_source(
        args.language_model,
        DEFAULT_LANGUAGE_MODEL,
        HF_LANGUAGE_MODEL,
        args.local_files_only,
    )
    backbone_checkpoint = resolve_checkpoint(
        args.backbone_checkpoint, args.local_files_only
    )
    model, image_processor, tokenizer = create_model(
        vision_encoder_path=vision_encoder,
        lang_encoder_path=language_model,
        tokenizer_path=language_model,
        checkpoint_path=backbone_checkpoint,
        use_local_files=args.local_files_only,
    )
    trainable = configure_trainable_parameters(model)
    model.to(device)

    full_train = CalvinDataset(args.dataset, "training", args.window_size)
    train_size = int(len(full_train) * 0.8)
    holdout_size = len(full_train) - train_size
    train_set, holdout_set = random_split(
        full_train,
        [train_size, holdout_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    official_val = CalvinDataset(args.dataset, "validation", args.window_size)

    train_loader = build_loader(
        train_set, image_processor, tokenizer, args.batch_size, True, args.workers
    )
    holdout_loader = build_loader(
        holdout_set, image_processor, tokenizer, args.batch_size, False, args.workers
    )
    official_val_loader = build_loader(
        official_val, image_processor, tokenizer, args.batch_size, False, args.workers
    )

    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def schedule(step):
        if step < args.warmup_steps:
            return float(step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            args.steps - args.warmup_steps - 1, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    iterator = iter(train_loader)
    best_holdout = float("inf")

    print(f"device: {device}")
    print(f"train windows: {len(train_set)}")
    print(f"holdout windows: {len(holdout_set)}")
    print(f"official validation windows: {len(official_val)}")
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,}")

    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(
            batch["rgb_static"].to(device, non_blocking=True),
            batch["input_ids"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True),
        )
        loss, pose_loss, gripper_loss = action_loss(
            prediction, batch["actions"].to(device, non_blocking=True)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"step {step + 1}/{args.steps} "
                f"loss={loss.item():.5f} pose={pose_loss.item():.5f} "
                f"gripper={gripper_loss.item():.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e}"
            )

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            holdout_loss, holdout_pose, holdout_gripper = evaluate(
                model, holdout_loader, device
            )
            print(
                f"holdout step {step + 1}: loss={holdout_loss:.5f} "
                f"pose={holdout_pose:.5f} gripper={holdout_gripper:.5f}"
            )
            if holdout_loss < best_holdout:
                best_holdout = holdout_loss
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "holdout_loss": best_holdout,
                    },
                    args.output / "best.pt",
                )

    final_loss = evaluate(model, official_val_loader, device)
    print(
        f"official validation: loss={final_loss[0]:.5f} "
        f"pose={final_loss[1]:.5f} gripper={final_loss[2]:.5f}"
    )
    torch.save(
        {"step": args.steps, "model_state_dict": model.state_dict()},
        args.output / "last.pt",
    )


if __name__ == "__main__":
    main()
