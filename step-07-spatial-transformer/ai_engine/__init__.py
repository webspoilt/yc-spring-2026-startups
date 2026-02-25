"""
Step 7: Spatial Transformer AI Engine
=====================================
PyTorch-based 3D point cloud processing pipeline:
- PointNet++ encoder for hierarchical feature learning
- Spatial Attention with Euclidean distance bias
- Quaternion prediction for rigid-body transformations
"""

from .models.pointnet import PointNetPlusPlusEncoder
from .models.spatial_attention import SpatialAttentionModule
from .models.spatial_transformer import SpatialTransformerNetwork, PointNetPlusPlus, SpatialTransformer

__all__ = [
    "PointNetPlusPlusEncoder",
    "SpatialAttentionModule",
    "SpatialTransformerNetwork",
    "PointNetPlusPlus",
    "SpatialTransformer",
]
