"""Self-trained AudioEncoder neural network module.
Random initialization from scratch. 2D ResNet + temporal attention pooling.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .config import DIM
except ImportError:
    from config import DIM


class ResBlock(nn.Module):
    """2D Convolutional residual block with batch normalization and GELU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + res)


class AudioEncoder(nn.Module):
    """Deep 2D Convolutional network with temporal attention pooling for audio log-mel spectrograms.
    Trained strictly from scratch from random initialization.
    Outputs unit-normalized 128D audio embeddings in the shared coordinate space.
    """
    def __init__(self, dim: int = DIM, n_mels: int = 128):
        super().__init__()
        self.dim = dim
        self.n_mels = n_mels

        # Front-end convolutional stem
        self.conv_in = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Residual convolutional stages
        self.res1 = ResBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.res2 = ResBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.res3 = ResBlock(128, 128)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool frequency axis to 1

        # Temporal attention pooling over frame representations
        self.attn = nn.Linear(128, 1)

        # Projection head to shared 128D vector space
        self.proj = nn.Linear(128, dim)
        self.norm = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        """Random initialization for all trainable layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_segment(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single log-mel spectrogram segment [B, 1, 128, T] -> [B, dim]."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = F.gelu(self.bn_in(self.conv_in(x)))
        h = self.pool1(h)
        h = self.pool2(self.res1(h))
        h = self.pool3(self.res2(h))
        h = self.freq_pool(self.res3(h)).squeeze(2)  # [B, 128, T']

        # Temporal attention pooling
        t = h.transpose(1, 2)  # [B, T', 128]
        weights = F.softmax(self.attn(t), dim=1)  # [B, T', 1]
        pooled = (t * weights).sum(dim=1)  # [B, 128]

        out = self.norm(self.proj(pooled))
        return F.normalize(out, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single segment or multi-segment stack.
        If [B, 1, 128, T], returns [B, dim].
        If [B, num_segments, 1, 128, T], pools across segments and returns [B, dim].
        """
        if x.dim() == 4:
            return self.forward_segment(x)
        elif x.dim() == 5:
            # [B, num_segments, 1, 128, T]
            b, num_seg, c, n_mels, t_frames = x.shape
            flat_x = x.view(b * num_seg, c, n_mels, t_frames)
            flat_emb = self.forward_segment(flat_x)  # [B * num_seg, dim]
            seg_emb = flat_emb.view(b, num_seg, self.dim)  # [B, num_seg, dim]
            # Mean pool across multi-segments and renormalize
            track_emb = F.normalize(seg_emb.mean(dim=1), dim=-1)
            return track_emb
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")

    def forward_multiview(self, segments_a: torch.Tensor, segments_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode two views/segments of the same tracks for self-supervised contrastive learning."""
        z_a = self.forward_segment(segments_a)
        z_b = self.forward_segment(segments_b)
        return z_a, z_b
