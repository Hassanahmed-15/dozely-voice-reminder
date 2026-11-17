"""A compact Conformer acoustic model, initialized randomly (no pretrained weights)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsample(nn.Module):
    """Reduce time by 4 and project the mel bands to the model dimension."""

    def __init__(self, n_mels: int, dim: int, channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.proj = nn.Linear(channels * math.ceil(math.ceil(n_mels / 2) / 2), dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self.conv(x.unsqueeze(1)).permute(0, 2, 1, 3).flatten(2)
        return self.proj(x), ((lengths + 3) // 4).clamp(max=x.size(1))


class FeedForward(nn.Sequential):
    def __init__(self, dim: int, dropout: float):
        super().__init__(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout),
                         nn.Linear(dim * 4, dim), nn.Dropout(dropout))


class ConvModule(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1), nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
            nn.GroupNorm(1, dim), nn.SiLU(), nn.Conv1d(dim, dim, 1), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(self.norm(x).transpose(1, 2)).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, kernel: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.conv = ConvModule(dim, kernel, dropout)
        self.ff2 = FeedForward(dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        y = self.attn_norm(x)
        x = x + self.drop(self.attn(y, y, y, key_padding_mask=pad_mask, need_weights=False)[0])
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class SpeechModel(nn.Module):
    def __init__(self, n_tokens: int, n_mels: int = 80, dim: int = 144, blocks: int = 5, heads: int = 4,
                 kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.config = dict(n_tokens=n_tokens, n_mels=n_mels, dim=dim, blocks=blocks, heads=heads, kernel=kernel, dropout=dropout)
        self.sub = ConvSubsample(n_mels, dim)
        self.pos = nn.Parameter(torch.zeros(1, 4000, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(ConformerBlock(dim, heads, kernel, dropout) for _ in range(blocks))
        self.out = nn.Linear(dim, n_tokens)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x, lengths = self.sub(x, lengths)
        x = self.drop(x + self.pos[:, : x.size(1)])
        mask = torch.arange(x.size(1), device=x.device)[None] >= lengths[:, None]
        for block in self.blocks:
            x = block(x, mask)
        return F.log_softmax(self.out(x), -1), lengths
