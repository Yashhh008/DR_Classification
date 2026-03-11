"""
Transformer Encoder blocks for the hybrid architecture.
Implements Multi-Head Self-Attention and a stack of Transformer Encoder layers.
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert CNN feature-map spatial positions into a sequence of tokens
    with positional embeddings.

    Input : (B, C, H, W) feature map from CNN
    Output: (B, N, embed_dim)  where N = H * W
    """

    def __init__(self, in_channels: int, embed_dim: int, num_patches: int):
        super().__init__()
        self.projection = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS token
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, C, H, W) → (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.projection(x)  # (B, N, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)

        # Add positional embedding (interpolate if size mismatch)
        if x.size(1) != self.position_embedding.size(1):
            pos = self._interpolate_pos(x.size(1))
        else:
            pos = self.position_embedding
        x = x + pos
        return x

    def _interpolate_pos(self, target_len: int) -> torch.Tensor:
        pos = self.position_embedding  # (1, L, D)
        pos = pos.transpose(1, 2)  # (1, D, L)
        pos = nn.functional.interpolate(pos, size=target_len,
                                        mode="linear", align_corners=False)
        return pos.transpose(1, 2)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder layer with pre-norm."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder layers."""

    def __init__(self, in_channels: int, embed_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4,
                 mlp_dim: int = 512, dropout: float = 0.1,
                 num_patches: int = 49):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, num_patches)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CNN feature map (B, C, H, W)
        Returns:
            CLS token output (B, embed_dim)
        """
        x = self.patch_embed(x)       # (B, N+1, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 0]                # CLS token → (B, embed_dim)
