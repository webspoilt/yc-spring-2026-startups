from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
import random

router = APIRouter()


class GuidanceRequest(BaseModel):
    pose_type: str
    form_score: float
    issues: List[dict]


class GuidanceResponse(BaseModel):
    audio_text: str
    priority: str
    suggestions: List[str]


@router.post("/generate/guidance", response_model=GuidanceResponse)
async def generate_guidance(request: GuidanceRequest):
    """Generate audio guidance based on pose analysis."""
    
    suggestions = []
    priority = "info"
    
    if request.form_score < 50:
        priority = "critical"
        suggestions.append("Stop immediately and rest")
    elif request.form_score < 70:
        priority = "warning"
        suggestions.append("Focus on form over weight")
    
    for issue in request.issues:
        if issue.get("severity") == "moderate":
            suggestions.append(f"Fix your {issue.get('joint')} position")
    
    pose_guidance = {
        "squat": "Lower your hips to parallel",
        "lunge": "Keep your torso upright",
        "plank": "Engage your core and glutes",
        "standing": "Stand tall with shoulders back"
    }
    
    audio_text = pose_guidance.get(request.pose_type, "Continue your exercise")
    
    if suggestions:
        audio_text += ". " + " ".join(suggestions[:2])
    
    return GuidanceResponse(
        audio_text=audio_text,
        priority=priority,
        suggestions=suggestions
    )


@router.get("/exercises")
async def list_exercises():
    """List supported exercises."""
    return {
        "exercises": [
            {"name": "squat", "description": "Barbell or bodyweight squat"},
            {"name": "lunge", "description": "Forward or reverse lunge"},
            {"name": "plank", "description": "Core stabilization exercise"},
            {"name": "deadlift", "description": "Conventional or sumo deadlift"},
            {"name": "pushup", "description": "Standard pushup"}        ]
    }


@router.get("/audio/voices")
async def get_voice_options():
    """Get available TTS voices."""
    return {
        "voices": [
            {"id": "female_1", "name": "Sarah", "gender": "female"},
            {"id": "male_1", "name": "James", "gender": "male"},
            {"id": "neutral", "name": "Alex", "gender": "neutral"}
        ],
        "default": "female_1"
    }
