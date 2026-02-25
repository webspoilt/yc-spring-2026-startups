"""
Spatial Transformer Network — End-to-End 3D Transformation Predictor.

Combines PointNet++ encoding, Spatial Attention with Euclidean bias, and a
Quaternion prediction head for estimating rigid-body transformations between
point cloud pairs.

Pipeline:
  Source + Target point clouds → PointNet++ encode each → Cross-attend →
  Spatial Attention refinement → Quaternion + Translation prediction

Output: Unit quaternion (w, x, y, z) + translation vector (tx, ty, tz)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from .pointnet import PointNetPlusPlusEncoder
from .spatial_attention import SpatialAttentionModule


class QuaternionHead(nn.Module):
    """
    Predicts a unit quaternion (w, x, y, z) and translation (tx, ty, tz)
    from a feature vector. Uses L2 normalization to enforce unit quaternion
    constraint.
    """

    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Quaternion branch: predict (w, x, y, z)
        self.quat_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

        # Translation branch: predict (tx, ty, tz)
        self.trans_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        # Initialize quaternion head bias to identity rotation [1, 0, 0, 0]
        nn.init.zeros_(self.quat_head[-1].weight)
        self.quat_head[-1].bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_dim) combined feature vector

        Returns:
            quaternion: (B, 4) unit quaternion [w, x, y, z]
            translation: (B, 3) translation [tx, ty, tz]
        """
        shared_feat = self.shared(x)

        quat = self.quat_head(shared_feat)
        quat = F.normalize(quat, p=2, dim=-1)  # enforce ||q|| = 1

        translation = self.trans_head(shared_feat)

        return quat, translation


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer to fuse source and target point cloud features.
    The source queries attend to the target keys/values.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: (B, 1, D) source global feature
            target: (B, 1, D) target global feature

        Returns:
            fused: (B, D) cross-attended fused feature
        """
        # Cross attention: source attends to target
        residual = source
        source_normed = self.norm1(source)
        target_normed = self.norm1(target)
        attn_out, _ = self.mha(source_normed, target_normed, target_normed)
        source = residual + attn_out

        # FFN
        source = source + self.ffn(self.norm2(source))

        return source.squeeze(1)  # (B, D)


class SpatialTransformerNetwork(nn.Module):
    """
    Full Spatial Transformer: encodes two point clouds, applies spatial
    attention with Euclidean bias, cross-attends, and predicts a rigid
    transformation (quaternion + translation).

    Args:
        in_channels: per-point input dimension (default 3 for xyz)
        latent_dim: PointNet++ output dimension
        attn_heads: number of attention heads
        attn_layers: number of spatial attention layers
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256,
                 attn_heads: int = 8, attn_layers: int = 4):
        super().__init__()

        self.encoder = PointNetPlusPlusEncoder(in_channels, latent_dim)

        # Spatial attention for refining encoded features
        self.spatial_attn = SpatialAttentionModule(
            embed_dim=latent_dim, num_heads=attn_heads,
            num_layers=attn_layers, dropout=0.1
        )

        # Cross-attention for source-target fusion
        self.cross_attn = CrossAttentionFusion(latent_dim, attn_heads)

        # Quaternion prediction from fused features
        self.quat_predictor = QuaternionHead(latent_dim * 2)

    def encode_cloud(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode a single point cloud to latent space."""
        return self.encoder(xyz)  # (B, latent_dim)

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> dict:
        """
        Args:
            source: (B, N, 3) source point cloud
            target: (B, M, 3) target point cloud

        Returns:
            dict with keys:
                quaternion: (B, 4) predicted rotation
                translation: (B, 3) predicted translation
                rotation_matrix: (B, 3, 3) derived rotation matrix
                source_feat: (B, D) source embedding
                target_feat: (B, D) target embedding
        """
        # Encode both point clouds
        src_feat = self.encode_cloud(source)  # (B, latent_dim)
        tgt_feat = self.encode_cloud(target)  # (B, latent_dim)

        # Cross-attend (unsqueeze for sequence dim)
        fused_src = self.cross_attn(src_feat.unsqueeze(1), tgt_feat.unsqueeze(1))  # (B, D)

        # Concatenate for quaternion head
        combined = torch.cat([fused_src, tgt_feat], dim=-1)  # (B, 2*D)

        # Predict quaternion + translation
        quaternion, translation = self.quat_predictor(combined)

        # Derive rotation matrix from quaternion
        rot_matrix = quaternion_to_rotation_matrix(quaternion)

        return {
            "quaternion": quaternion,
            "translation": translation,
            "rotation_matrix": rot_matrix,
            "source_feat": src_feat,
            "target_feat": tgt_feat,
        }


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion (w, x, y, z) to 3×3 rotation matrix.

    Uses the formula:
        R = I + 2w*[v]_x + 2*[v]_x^2
    where [v]_x is the skew-symmetric matrix of the imaginary part.

    Args:
        q: (B, 4) quaternion [w, x, y, z]

    Returns:
        R: (B, 3, 3) rotation matrix
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    B = q.shape[0]
    R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def geodesic_loss(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """
    Geodesic loss for quaternion regression.

    L = arccos(|<q_pred, q_target>|) — invariant to quaternion sign ambiguity.

    Args:
        q_pred: (B, 4) predicted quaternion
        q_target: (B, 4) ground truth quaternion

    Returns:
        loss: scalar geodesic loss
    """
    dot = torch.abs(torch.sum(q_pred * q_target, dim=-1))
    dot = torch.clamp(dot, 0.0, 1.0)
    loss = torch.acos(dot)
    return loss.mean()


class TransformationLoss(nn.Module):
    """
    Combined loss for transformation prediction:
        L = alpha * geodesic_loss(q) + beta * MSE(translation)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: dict, target_quat: torch.Tensor,
                target_trans: torch.Tensor) -> dict:
        q_loss = geodesic_loss(pred["quaternion"], target_quat)
        t_loss = F.mse_loss(pred["translation"], target_trans)
        total = self.alpha * q_loss + self.beta * t_loss
        return {
            "total": total,
            "quaternion_loss": q_loss,
            "translation_loss": t_loss,
        }


# ─────────────────────────────────────────────────────────────
# Legacy compatibility wrappers (used by the existing router)
# ─────────────────────────────────────────────────────────────

class PointNetPlusPlus:
    """Stateless wrapper for the /encode API endpoint."""

    def __init__(self):
        self._model = PointNetPlusPlusEncoder(in_channels=3, latent_dim=256)
        self._model.eval()

    def encode(self, points_list: list) -> np.ndarray:
        with torch.no_grad():
            pts = torch.tensor(points_list, dtype=torch.float32).unsqueeze(0)
            out = self._model(pts)
            return out.squeeze(0).cpu().numpy()


class SpatialTransformer:
    """Stateless wrapper for the /predict API endpoint."""

    def __init__(self):
        self._model = SpatialTransformerNetwork(in_channels=3, latent_dim=256)
        self._model.eval()

    def predict(self, source: list, target: list, query: list = None) -> list:
        with torch.no_grad():
            src = torch.tensor(source, dtype=torch.float32).unsqueeze(0)
            tgt = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            result = self._model(src, tgt)
            return result["quaternion"].squeeze(0).cpu().numpy().tolist()

    @staticmethod
    def quaternion_to_matrix(q: list) -> list:
        qt = torch.tensor([q], dtype=torch.float32)
        R = quaternion_to_rotation_matrix(qt)
        return R.squeeze(0).cpu().numpy().tolist()
