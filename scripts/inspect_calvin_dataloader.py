#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer, CLIPImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboflamingo import create_calvin_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("calvin/dataset/calvin_debug_dataset"),
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
        "--tokenizer",
        type=Path,
        default=Path("models/language_conditioned/mpt_1b_dolly"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=32)
    args = parser.parse_args()

    image_processor = CLIPImageProcessor.from_pretrained(
        args.vision_encoder, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    loader = create_calvin_dataloader(
        args.dataset,
        image_processor,
        tokenizer,
        batch_size=args.batch_size,
        window_size=args.window_size,
        shuffle=False,
    )
    batch = next(iter(loader))

    print(f"dataset samples: {len(loader.dataset)}")
    for key in ("rgb_static", "rgb_gripper", "input_ids", "actions", "robot_obs"):
        print(f"{key}: {tuple(batch[key].shape)} {batch[key].dtype}")
    print(f"instruction: {batch['instructions'][0]}")
    print(f"task: {batch['tasks'][0]}")
    print(f"start frame: {batch['start_frames'][0].item()}")
    print(f"first action: {batch['actions'][0, 0].tolist()}")


if __name__ == "__main__":
    main()
