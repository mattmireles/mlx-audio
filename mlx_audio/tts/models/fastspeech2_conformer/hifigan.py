
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

@dataclass
class FastSpeech2ConformerHifiGanConfig:
    model_in_dim: int = 80
    upsample_initial_channel: int = 512
    upsample_rates: list = [8, 8, 2, 2]
    upsample_kernel_sizes: list = [16, 16, 4, 4]
    resblock_kernel_sizes: list = [3, 7, 11]
    resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    leaky_relu_slope: float = 0.1
    normalize_before: bool = True

# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]

        self.convs2 = [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def __call__(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states

class ConformerHifiGan(nn.Module):


    def __init__(self, config: FastSpeech2ConformerHifiGanConfig):
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = []
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = []
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        self.mean = mx.zeros(config.model_in_dim)
        self.scale = mx.ones(config.model_in_dim)



    def __call__(self, spectrogram: mx.array) -> mx.array:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        if self.config.normalize_before:
            spectrogram = (spectrogram - self.mean) / self.scale

        is_batched = spectrogram.dim() == 3
        if not is_batched:
            spectrogram = mx.expand_dims(spectrogram, 0)

        hidden_states = spectrogram.transpose(0, 2, 1)

        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = nn.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = mx.tanh(hidden_states)

        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            waveform = hidden_states.squeeze(0).transpose(1, 0).reshape(-1)
        else:
            # remove seq-len dim since this collapses to 1
            waveform = hidden_states.squeeze(1)

        return waveform
