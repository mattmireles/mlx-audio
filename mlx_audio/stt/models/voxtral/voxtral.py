import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig, ModelConfig, TextConfig


class Attention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = config.d_model // config.encoder_attention_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        bsz, tgt_len, _ = x.shape

        query_states = self.q_proj(x) * self.scaling
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=1.0, mask=attention_mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, tgt_len, self.embed_dim
        )

        return self.out_proj(attn_output)


class VoxtralEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        r = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = r + x

        r = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = r + x

        return x


class Encoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.layers = [
            VoxtralEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:

        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        x = x.transpose(0, 2, 1)
        embed_pos = self.embed_positions.weight

        x = (x + embed_pos).astype(x.dtype)

        for encoder_layer in self.layers:
            x = encoder_layer(x, mask)

        return self.layer_norm(x)


class MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.intermediate_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def __call__(self, audio_features: mx.array) -> mx.array:
        hidden_states = self.linear_1(audio_features)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.text_config.vocab_size

        self.audio_tower = Encoder(config.audio_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_audio_embeds(self, x: mx.array) -> mx.array:
        audio_embeds = self.audio_tower(x).reshape(
            -1, self.config.audio_config.intermediate_size
        )
        audio_embeds = self.multi_modal_projector(audio_embeds)
        return audio_embeds

    def __call__(
        self,
        input_ids: mx.array,
        input_features: mx.array,
    ) -> mx.array:

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = None

        if input_features is not None:
            audio_embeds = self.get_audio_embeds(input_features)

            if inputs_embeds is not None:
                # Replace audio token placeholders with audio embeddings
                audio_token_mask = input_ids == self.config.audio_token_id
                inputs_embeds = mx.where(
                    audio_token_mask[..., None], audio_embeds, inputs_embeds
                )
            else:
                inputs_embeds = audio_embeds

        logits = self.lm_head(inputs_embeds)

        return logits

    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None):
        if config is None:
            import json

            with open(f"{model_path}/config.json", "r") as f:
                config_dict = json.load(f)
            config = ModelConfig.from_dict(config_dict)

        model = cls(config)

        # Load weights
        weights = mx.load(f"{model_path}/model.safetensors")
        model.load_weights(list(weights.items()))

        return model
