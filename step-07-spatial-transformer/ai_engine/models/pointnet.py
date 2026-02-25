"""
PointNet++ Encoder for Hierarchical Point Cloud Feature Learning.

Implements Set Abstraction (SA) layers with multi-scale grouping,
following the architecture from Qi et al. (2017) "PointNet++: Deep
Hierarchical Feature Learning on Point Sets in a Metric Space".

Architecture:
  Input N×3 point cloud → SA1 (512 points) → SA2 (128 points) → SA3 (1 global)
  Each SA layer: Sampling → Grouping → PointNet-style MLP → Max pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance between two point sets.

    Args:
        src: (B, N, C) source points
        dst: (B, M, C) target points

    Returns:
        dist: (B, N, M) squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2.0 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(-2)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Iterative Farthest Point Sampling (FPS).

    Selects `npoint` points from the input such that each successive
    point is the one farthest from the already-selected set.

    Args:
        xyz: (B, N, 3) input point positions
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by index.

    Args:
        points: (B, N, C) input features
        idx: (B, S) or (B, S, K) index tensor

    Returns:
        gathered: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor,
                     new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query: find all points within a radius of each centroid.

    Args:
        radius: search radius
        nsample: maximum number of neighbors per centroid
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) centroid points

    Returns:
        group_idx: (B, S, nsample) indices of grouped points
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class SharedMLP(nn.Module):
    """Shared MLP applied independently to each point (1×1 convolution)."""

    def __init__(self, in_channels: int, out_channels_list: List[int],
                 bn: bool = True, activation: str = "relu"):
        super().__init__()
        layers = []
        for out_ch in out_channels_list:
            layers.append(nn.Conv1d(in_channels, out_ch, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_ch))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            in_channels = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N) → (B, C_out, N)"""
        return self.mlp(x)


class SetAbstractionLayer(nn.Module):
    """
    PointNet++ Set Abstraction (SA) Layer.

    Combines FPS downsampling, ball query grouping, and a shared MLP
    followed by max pooling to produce a compact representation.

    Args:
        npoint: number of centroids to sample
        radius: ball query search radius
        nsample: max neighbors per centroid
        in_channel: input feature dimension (including xyz=3)
        mlp_channels: list of output channels for the shared MLP
    """

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp_channels: List[int]):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = SharedMLP(in_channel, mlp_channels)

    def forward(self, xyz: torch.Tensor,
                points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) input positions
            points: (B, N, D) optional input features

        Returns:
            new_xyz: (B, npoint, 3) sampled centroid positions
            new_points: (B, npoint, D') abstracted features
        """
        B, N, C = xyz.shape

        # Farthest point sampling
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        # Ball query grouping
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)

        if points is not None:
            grouped_points = index_points(points, idx)
            grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz_norm

        # Shared MLP: (B, S, K, D) → (B, D, S*K) → (B, D', S*K) → reshape
        grouped_points = grouped_points.view(B, self.npoint * self.nsample, -1)
        grouped_points = grouped_points.permute(0, 2, 1)  # (B, D, S*K)
        grouped_points = self.mlp(grouped_points)  # (B, D', S*K)
        grouped_points = grouped_points.view(B, -1, self.npoint, self.nsample)

        # Max Pool over neighbors
        new_points = torch.max(grouped_points, dim=-1)[0]  # (B, D', S)
        new_points = new_points.permute(0, 2, 1)  # (B, S, D')

        return new_xyz, new_points


class GlobalSetAbstraction(nn.Module):
    """
    Global SA layer: aggregates ALL points into a single global descriptor
    via shared MLP + max pooling (no sampling/grouping).
    """

    def __init__(self, in_channel: int, mlp_channels: List[int]):
        super().__init__()
        self.mlp = SharedMLP(in_channel, mlp_channels)

    def forward(self, xyz: torch.Tensor,
                points: Optional[torch.Tensor] = None) -> Tuple[None, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, D)

        Returns:
            None (no new xyz)
            new_points: (B, 1, D') global feature
        """
        if points is not None:
            combined = torch.cat([xyz, points], dim=-1)  # (B, N, 3+D)
        else:
            combined = xyz

        combined = combined.permute(0, 2, 1)  # (B, C, N)
        combined = self.mlp(combined)  # (B, D', N)
        new_points = torch.max(combined, dim=-1)[0]  # (B, D')
        new_points = new_points.unsqueeze(1)  # (B, 1, D')
        return None, new_points


class PointNetPlusPlusEncoder(nn.Module):
    """
    Full PointNet++ Encoder: hierarchical feature extraction from raw point clouds.

    Pipeline:
        Input (B, N, 3) → SA1 (512 pts, 64-dim) → SA2 (128 pts, 128-dim)
                        → GlobalSA (1 pt, 1024-dim) → FC → latent_dim output

    Args:
        in_channels: input feature dimension per point (default=3 for xyz)
        latent_dim: output embedding dimension (default=256)
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # SA1: N → 512 points, radius 0.2, 32 neighbors
        self.sa1 = SetAbstractionLayer(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channels, mlp_channels=[64, 64, 128]
        )

        # SA2: 512 → 128 points, radius 0.4, 64 neighbors
        self.sa2 = SetAbstractionLayer(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp_channels=[128, 128, 256]
        )

        # SA3 (global): 128 → 1 global descriptor
        self.sa3 = GlobalSetAbstraction(
            in_channel=256 + 3, mlp_channels=[256, 512, 1024]
        )

        # Final projection to latent space
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) raw point cloud coordinates

        Returns:
            latent: (B, latent_dim) point cloud embedding
        """
        # Hierarchical abstraction
        new_xyz, new_points = self.sa1(xyz, None)
        new_xyz, new_points = self.sa2(new_xyz, new_points)
        _, new_points = self.sa3(new_xyz, new_points)

        # Flatten and project
        latent = new_points.squeeze(1)  # (B, 1024)
        latent = self.fc(latent)  # (B, latent_dim)
        return latent

    def encode(self, points_list: list) -> dict:
        """Convenience method for API usage with raw Python lists."""
        self.eval()
        with torch.no_grad():
            pts = torch.tensor(points_list, dtype=torch.float32).unsqueeze(0)
            embedding = self.forward(pts)
            return {
                "embedding": embedding.squeeze(0).cpu().numpy().tolist(),
                "latent_dim": self.latent_dim,
            }
