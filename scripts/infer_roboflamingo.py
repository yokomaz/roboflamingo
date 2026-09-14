#!/usr/bin/env python
"""Run one-step inference with the assembled RoboFlamingo model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboflamingo import create_model


def load_image(image_path, frame_path, image_processor):
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.fromarray(np.load(frame_path)["rgb_static"])
    processed = image_processor(images=image, return_tensors="pt")["pixel_values"]
    return processed.squeeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=Path(
            "models/language_conditioned/"
            "openflamingo_3b_vitl_mpt1b_langinstruct/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--vision-encoder",
        type=Path,
        default=Path(
            "models/language_conditioned/clip_vit_l14/"
            "modelscope_clip_vit_large_patch14"
        ),
    )
    parser.add_argument(
        "--language-model",
        type=Path,
        default=Path("models/language_conditioned/mpt_1b_dolly"),
    )
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--frame",
        type=Path,
        default=Path(
            "calvin/dataset/calvin_debug_dataset/"
            "training/episode_0358482.npz"
        ),
    )
    parser.add_argument("--instruction", default="Describe the robot scene.")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/cache"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.backbone_checkpoint.is_file():
        raise FileNotFoundError(args.backbone_checkpoint)
    if not args.vision_encoder.is_dir():
        raise FileNotFoundError(args.vision_encoder)
    if not args.language_model.is_dir():
        raise FileNotFoundError(args.language_model)
    if args.image is None and not args.frame.is_file():
        raise FileNotFoundError(args.frame)
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(args.image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor, tokenizer = create_model(
        vision_encoder_path=str(args.vision_encoder),
        lang_encoder_path=str(args.language_model),
        tokenizer_path=str(args.language_model),
        checkpoint_path=str(args.backbone_checkpoint),
        cache_dir=str(args.cache_dir),
        use_local_files=args.local_files_only,
    )

    if args.policy_checkpoint:
        state = torch.load(
            args.policy_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.policy_head.load_state_dict(state, strict=True)
    else:
        print("warning: policy head is randomly initialized")

    print(f"Loaded model from {args.backbone_checkpoint}")
    model = model.to(device).eval()
    vision_x = load_image(args.image, args.frame, image_processor).to(device)

    tokenizer.padding_side = "left"
    tokens = tokenizer(
        [f"<image>{args.instruction}<|endofchunk|>"],
        return_tensors="pt",
    )
    lang_x = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device).bool()

    with torch.inference_mode():
        action, _ = model(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
        )

    print(f"device: {device}")
    print(f"image shape: {tuple(vision_x.shape)}")
    print(f"action shape: {tuple(action.shape)}")
    print(f"action: {action[0].detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
