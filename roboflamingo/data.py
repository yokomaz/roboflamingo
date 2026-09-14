from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CalvinDataset(Dataset):
    """Load fixed-length, language-conditioned windows from a CALVIN split."""

    def __init__(
        self,
        dataset_root,
        split="training",
        window_size=32,
        action_key="rel_actions",
        stride=1,
    ):
        self.split_dir = Path(dataset_root) / split
        self.window_size = window_size
        self.action_key = action_key

        annotation_path = self.split_dir / "lang_annotations" / "auto_lang_ann.npy"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)

        annotations = np.load(annotation_path, allow_pickle=True).item()
        self.instructions = annotations["language"]["ann"]
        self.tasks = annotations["language"]["task"]
        self.samples = []
        for annotation_index, (start, end) in enumerate(annotations["info"]["indx"]):
            stop = int(end) + 1 - window_size
            self.samples.extend(
                (frame, annotation_index)
                for frame in range(int(start), stop, stride)
            )

        if not self.samples:
            raise ValueError(
                f"No {window_size}-frame language windows found in {self.split_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        start, annotation_index = self.samples[index]
        keys = ("rgb_static", "rgb_gripper", "robot_obs", self.action_key)
        values = {key: [] for key in keys}
        for frame in range(start, start + self.window_size):
            path = self.split_dir / f"episode_{frame:07d}.npz"
            with np.load(path) as item:
                for key in keys:
                    values[key].append(item[key])

        values = {key: np.stack(items) for key, items in values.items()}

        return {
            "rgb_static": values["rgb_static"],
            "rgb_gripper": values["rgb_gripper"],
            "robot_obs": torch.from_numpy(values["robot_obs"]).float(),
            "actions": torch.from_numpy(values[self.action_key]).float(),
            "instruction": str(self.instructions[annotation_index]),
            "task": str(self.tasks[annotation_index]),
            "start_frame": start,
        }


class CalvinCollator:
    def __init__(self, image_processor, tokenizer, max_text_length=32):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, samples):
        batch_size = len(samples)
        window_size = len(samples[0]["rgb_static"])

        def process_images(key):
            images = np.concatenate([sample[key] for sample in samples])
            pixels = self.image_processor(
                images=list(images), return_tensors="pt"
            )["pixel_values"]
            return pixels.reshape(batch_size, window_size, *pixels.shape[1:])

        texts = [
            f'<image>{sample["instruction"].strip()}<|endofchunk|>'
            for sample in samples
        ]
        tokens = self.tokenizer(
            texts,
            max_length=self.max_text_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "rgb_static": process_images("rgb_static"),
            "rgb_gripper": process_images("rgb_gripper"),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"].bool(),
            "actions": torch.stack([sample["actions"] for sample in samples]),
            "robot_obs": torch.stack([sample["robot_obs"] for sample in samples]),
            "instructions": [sample["instruction"] for sample in samples],
            "tasks": [sample["task"] for sample in samples],
            "start_frames": torch.tensor(
                [sample["start_frame"] for sample in samples]
            ),
        }


def create_calvin_dataloader(
    dataset_root,
    image_processor,
    tokenizer,
    *,
    split="training",
    window_size=32,
    batch_size=1,
    action_key="rel_actions",
    stride=1,
    shuffle=True,
    num_workers=0,
):
    dataset = CalvinDataset(
        dataset_root,
        split=split,
        window_size=window_size,
        action_key=action_key,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=CalvinCollator(image_processor, tokenizer),
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
