from pathlib import Path
from typing import Optional

import torch
from torch import nn


class LSTMPolicyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, action_dim: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.pose = nn.Linear(hidden_dim, action_dim - 1)
        self.gripper = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden=None):
        pooled = features.max(dim=-2).values
        output, hidden = self.lstm(pooled, hidden)
        return torch.cat((self.pose(output), self.gripper(output)), dim=-1), hidden


class RoboFlamingo(nn.Module):
    def __init__(self, flamingo, policy_head):
        super().__init__()
        self.flamingo = flamingo
        self.policy_head = policy_head

    def encode(self, vision_x, lang_x, attention_mask=None):
        self.flamingo._encode_vision_x(vision_x)
        self.flamingo._condition_media_locations(lang_x)
        try:
            output = self.flamingo.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return output.hidden_states[-1]
        finally:
            self.flamingo.lang_encoder.clear_conditioned_layers()

    def forward(self, vision_x, lang_x, attention_mask=None, hidden=None):
        sequence_input = vision_x.ndim == 5
        if sequence_input:
            batch_size, sequence_length = vision_x.shape[:2]
            vision_x = vision_x.reshape(batch_size * sequence_length, *vision_x.shape[2:])
            vision_x = vision_x.unsqueeze(1).unsqueeze(2)
            lang_x = lang_x.unsqueeze(1).expand(-1, sequence_length, -1)
            lang_x = lang_x.reshape(batch_size * sequence_length, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).expand(-1, sequence_length, -1)
                attention_mask = attention_mask.reshape(batch_size * sequence_length, -1)
        features = self.encode(vision_x, lang_x, attention_mask)
        if sequence_input:
            features = features.reshape(batch_size, sequence_length, *features.shape[1:])
        return self.policy_head(features, hidden)


class HuggingFaceCLIPVision(nn.Module):
    """Adapt Transformers CLIP vision output to OpenFlamingo.s visual API."""

    def __init__(self, vision_model):
        super().__init__()
        self.visual = vision_model

    def forward(self, pixel_values):
        tokens = self.visual(pixel_values=pixel_values).last_hidden_state
        return None, tokens


def create_model(
    *,
    vision_encoder_path="ViT-L-14",
    vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
    tokenizer_path=None,
    checkpoint_path: Optional[str] = None,
    cross_attn_every_n_layers=1,
    policy_hidden_dim=1024,
    action_dim=7,
    cache_dir=None,
    use_local_files=False,
):
    vision_encoder_path = str(vision_encoder_path)
    if Path(vision_encoder_path).is_dir() or "/" in vision_encoder_path:
        return _create_model_from_huggingface_clip(
            vision_encoder_path=vision_encoder_path,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path or lang_encoder_path,
            checkpoint_path=checkpoint_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            policy_hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
            cache_dir=cache_dir,
            use_local_files=use_local_files,
        )

    from open_flamingo import create_model_and_transforms

    tokenizer_path = tokenizer_path or lang_encoder_path
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=vision_encoder_path,
        clip_vision_encoder_pretrained=vision_encoder_pretrained,
        lang_encoder_path=lang_encoder_path,
        tokenizer_path=tokenizer_path,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        cache_dir=cache_dir,
        use_local_files=use_local_files,
    )

    model = RoboFlamingo(
        flamingo=flamingo,
        policy_head=LSTMPolicyHead(
            input_dim=flamingo.lang_dim,
            hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
        ),
    )

    if checkpoint_path:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        model.flamingo.load_state_dict(checkpoint, strict=False)

    return model, image_processor, tokenizer


def _create_model_from_huggingface_clip(
    *,
    vision_encoder_path,
    lang_encoder_path,
    tokenizer_path,
    checkpoint_path,
    cross_attn_every_n_layers,
    policy_hidden_dim,
    action_dim,
    cache_dir,
    use_local_files,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
    from transformers import CLIPVisionModel
    from open_flamingo.src.flamingo import Flamingo
    from open_flamingo.src.flamingo_lm import FlamingoLMMixin
    from open_flamingo.src.utils import extend_instance

    vision_model = CLIPVisionModel.from_pretrained(
        vision_encoder_path, local_files_only=use_local_files
    )
    vision_encoder = HuggingFaceCLIPVision(vision_model)
    vision_container = nn.Module()
    vision_container.visual = vision_encoder
    image_processor = CLIPImageProcessor.from_pretrained(
        vision_encoder_path, local_files_only=use_local_files
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    class EmbeddingFnMixin:
        def get_input_embeddings(self):
            return self.transformer.wte

        def set_input_embeddings(self, new_embeddings):
            self.transformer.wte = new_embeddings

    extend_instance(lang_encoder, EmbeddingFnMixin)
    extend_instance(lang_encoder, FlamingoLMMixin)
    type(lang_encoder).get_input_embeddings = EmbeddingFnMixin.get_input_embeddings
    type(lang_encoder).set_input_embeddings = EmbeddingFnMixin.set_input_embeddings
    lang_encoder.set_decoder_layers_attr_name("transformer.blocks")
    lang_encoder.resize_token_embeddings(len(tokenizer))
    flamingo = Flamingo(
        vision_container,
        lang_encoder,
        tokenizer.encode("<|endofchunk|>")[-1],
        tokenizer.encode("<image>")[-1],
        vis_dim=vision_model.config.hidden_size,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
    )
    flamingo.requires_grad_(False)
    flamingo.perceiver.requires_grad_(True)
    flamingo.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    flamingo.lang_encoder.get_input_embeddings().requires_grad_(True)
    model = RoboFlamingo(
        flamingo=flamingo,
        policy_head=LSTMPolicyHead(
            input_dim=flamingo.lang_dim,
            hidden_dim=policy_hidden_dim,
            action_dim=action_dim,
        ),
    )
    if checkpoint_path:
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )
        model.flamingo.load_state_dict(checkpoint, strict=False)
    return model, image_processor, tokenizer
