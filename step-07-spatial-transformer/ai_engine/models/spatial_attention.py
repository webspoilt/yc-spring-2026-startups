"""
Spatial Attention Module with Euclidean Distance Bias.

Implements a multi-head attention mechanism where the attention scores
are biased by the Euclidean distance between 3D point positions,
encouraging the model to attend to spatially proximate features.

Architecture:
  Q, K, V projections → dot-product attention + distance bias → weighted sum
  Distance bias = -alpha * ||p_i - p_j||^2  (learnable alpha per head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class EuclideanDistanceBias(nn.Module):
    """
    Computes a learnable spatial bias based on Euclidean distances.

    For each pair of points (i, j), the bias is:
        bias(i,j) = -alpha_h * ||position_i - position_j||^2 + beta_h

    where alpha_h (> 0) and beta_h are per-head learnable parameters.
    The negative sign ensures closer points receive a positive attention boost.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(num_heads))
        self.beta = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 3) 3D coordinates

        Returns:
            bias: (B, num_heads, N, N) spatial attention bias
        """
        # Pairwise squared distances: (B, N, N)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)
        sq_dist = torch.sum(diff ** 2, dim=-1)  # (B, N, N)

        # Per-head scaling (alpha > 0 via softplus)
        alpha = F.softplus(self.log_alpha)  # (num_heads,)

        # Compute bias: (B, 1, N, N) * (1, H, 1, 1) → (B, H, N, N)
        bias = -alpha.view(1, self.num_heads, 1, 1) * sq_dist.unsqueeze(1)
        bias = bias + self.beta.view(1, self.num_heads, 1, 1)

        return bias


class SpatialMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Euclidean distance bias for 3D point features.

    Standard scaled dot-product attention is augmented with a spatial bias
    term that encodes geometric proximity:

        Attention(Q, K, V) = softmax(QK^T / sqrt(d) + DistBias) V

    Args:
        embed_dim: total feature dimension
        num_heads: number of attention heads
        dropout: attention dropout probability
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.distance_bias = EuclideanDistanceBias(num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, features: torch.Tensor, positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, N, D) point features
            positions: (B, N, 3) point 3D coordinates
            mask: (B, N) optional boolean mask (True = valid)

        Returns:
            output: (B, N, D) attended features
            attn_weights: (B, H, N, N) attention weights for visualization
        """
        B, N, D = features.shape

        # Linear projections → (B, N, H, d_h) → (B, H, N, d_h)
        Q = self.q_proj(features).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(features).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(features).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product + distance bias
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        spatial_bias = self.distance_bias(positions)  # (B, H, N, N)
        attn_scores = attn_scores + spatial_bias

        # Apply mask if provided
        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3)
            attn_scores = attn_scores.masked_fill(~mask_2d, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        context = torch.matmul(attn_weights, V)  # (B, H, N, d_h)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        output = self.out_proj(context)
        output = self.proj_dropout(output)

        return output, attn_weights


class SpatialAttentionBlock(nn.Module):
    """
    Transformer block with spatial attention, following Pre-LN convention.

    Structure: LayerNorm → SpatialMHA → Residual → LayerNorm → FFN → Residual
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SpatialMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * ffn_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor, positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN spatial attention
        normed = self.norm1(features)
        attn_out, attn_weights = self.attn(normed, positions, mask)
        features = features + attn_out

        # Pre-LN FFN
        features = features + self.ffn(self.norm2(features))

        return features, attn_weights


class SpatialAttentionModule(nn.Module):
    """
    Stack of Spatial Attention Blocks for point cloud feature refinement.

    Takes per-point features + 3D positions and runs them through L
    transformer blocks with Euclidean distance bias.

    Args:
        embed_dim: feature dimension
        num_heads: attention heads per block
        num_layers: number of stacked attention blocks
        dropout: dropout probability
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialAttentionBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Args:
            features: (B, N, D) input per-point features
            positions: (B, N, 3) point coordinates
            mask: (B, N) optional validity mask

        Returns:
            refined: (B, N, D) refined features
            all_attn: list of (B, H, N, N) from each layer
        """
        all_attn = []
        for layer in self.layers:
            features, attn_weights = layer(features, positions, mask)
            all_attn.append(attn_weights)
        features = self.final_norm(features)
        return features, all_attn
