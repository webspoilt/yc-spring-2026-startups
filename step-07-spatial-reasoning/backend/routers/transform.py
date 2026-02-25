from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

router = APIRouter()

class PointCloudInput(BaseModel):
    points: List[List[float]]
    num_points: int = 1024

class TransformRequest(BaseModel):
    source_points: List[List[float]]
    target_points: List[List[float]]
    query_points: List[List[float]]

@router.post("/encode")
async def encode_pointcloud(input: PointCloudInput):
    """Encode point cloud using PointNet++."""
    from ...ai_engine.models.spatial_transformer import PointNetPlusPlus
    model = PointNetPlusPlus()
    features = model.encode(input.points)
    return {"features": features, "shape": features.shape}

@router.post("/predict")
async def predict_transform(request: TransformRequest):
    """Predict transformation (quaternion) between point clouds."""
    from ...ai_engine.models.spatial_transformer import SpatialTransformer
    model = SpatialTransformer()
    quaternion = model.predict(request.source_points, request.target_points, request.query_points)
    return {"quaternion": quaternion, "rotation_matrix": model.quaternion_to_matrix(quaternion)}

@router.get("/models")
async def list_models():
    return {"models": ["PointNet++", "Spatial Transformer"], "status": "loaded"}
