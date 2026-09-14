#!/usr/bin/env python
"""Run one local OpenFlamingo inference using a CALVIN RGB frame."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/language_conditioned/openflamingo_3b_vitl_mpt1b_langinstruct/checkpoint.pt"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image file; otherwise use the first CALVIN rgb_static frame.",
    )
    parser.add_argument(
        "--calvin-frame",
        type=Path,
        default=Path("calvin/dataset/calvin_debug_dataset/training/episode_0358482.npz"),
    )
    parser.add_argument(
        "--prompt",
        default="Describe the robot scene.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    if args.image is None:
        if not args.calvin_frame.is_file():
            raise FileNotFoundError(args.calvin_frame)
        image = Image.fromarray(np.load(args.calvin_frame)["rgb_static"])
    else:
        image = Image.open(args.image).convert("RGB")

    from open_flamingo import create_model_and_transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        cross_attn_every_n_layers=1,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device).eval()
    vision_x = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0)
    vision_x = vision_x.to(device)

    tokenizer.padding_side = "left"
    lang_x = tokenizer(
        [f"<image>{args.prompt}<|endofchunk|>"],
        return_tensors="pt",
    )
    lang_x = {key: value.to(device) for key, value in lang_x.items()}

    with torch.inference_mode():
        generated = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
        )

    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
