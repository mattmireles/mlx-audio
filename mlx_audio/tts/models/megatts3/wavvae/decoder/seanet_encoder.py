

from typing import List

import torch
from torch import nn
from tts.modules.wavvae.encoder.common_modules.seanet import SEANetEncoder

class Encoder(nn.Module):
    def __init__(
        self,
        dowmsamples: List[int] = [6, 5, 5, 4, 2],
    ):
        super().__init__()

        # breakpoint()
        self.frame_rate = 25  # not use
        self.encoder = SEANetEncoder(causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
                                dimension=512, channels=1, n_filters=32, ratios=dowmsamples, activation='ELU',
                                kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
                                true_skip=False, compress=2)

    def forward(self, audio: torch.Tensor):
        audio = audio.unsqueeze(1)                  # audio(16,24000)
        emb = self.encoder(audio)
        return emb