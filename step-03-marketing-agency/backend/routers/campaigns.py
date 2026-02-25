from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from ...ai_engine.graph.campaign_workflow import CampaignWorkflow

router = APIRouter()

# Initialize workflow
workflow = CampaignWorkflow()

# In-memory storage
campaigns_db = {}


class CampaignCreate(BaseModel):
    name: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=10)
    target_audience: str
    tone: str = "professional"
    content_type: str = "blog_post"  # blog_post, social_media, email, ad


class CampaignStage(BaseModel):
    agent: str
    status: str  # pending, in_progress, completed, failed
    output: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class CampaignResponse(BaseModel):
    id: str
    name: str
    topic: str
    target_audience: str
    tone: str
    content_type: str
    status: str  # pending, running, completed, failed
    stages: List[CampaignStage]
    final_content: Optional[dict] = None
    created_at: datetime
    updated_at: datetime


@router.post("/campaigns", response_model=CampaignResponse)
async def create_campaign(campaign: CampaignCreate):
    """Create a new content generation campaign."""
    campaign_id = str(uuid.uuid4())
    
    now = datetime.now()
    db_campaign = {
        "id": campaign_id,
        "name": campaign.name,
        "topic": campaign.topic,
        "target_audience": campaign.target_audience,
        "tone": campaign.tone,
        "content_type": campaign.content_type,
        "status": "pending",
        "stages": [
            {"agent": "researcher", "status": "pending", "output": None},
            {"agent": "copywriter", "status": "pending", "output": None},
            {"agent": "designer", "status": "pending", "output": None},
            {"agent": "reviewer", "status": "pending", "output": None},
        ],
        "final_content": None,
        "created_at": now,
        "updated_at": now
    }
    
    campaigns_db[campaign_id] = db_campaign
    
    return CampaignResponse(**db_campaign)


@router.get("/campaigns", response_model=List[CampaignResponse])
async def list_campaigns():
    """List all campaigns."""
    return [CampaignResponse(**c) for c in campaigns_db.values()]


@router.get("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: str):
    """Get a specific campaign."""
    if campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return CampaignResponse(**campaigns_db[campaign_id])


@router.post("/campaigns/{campaign_id}/execute")
async def execute_campaign(campaign_id: str):
    """
    Execute the full content generation pipeline.
    Runs: Researcher -> Copywriter -> Designer -> Reviewer
    """
    if campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    campaign = campaigns_db[campaign_id]
    campaign["status"] = "running"
    campaign["updated_at"] = datetime.now()
    
    try:
        # Run the workflow
        result = await workflow.execute(
            topic=campaign["topic"],
            target_audience=campaign["target_audience"],
            tone=campaign["tone"],
            content_type=campaign["content_type"]
        )
        
        # Update stages with results
        for i, stage in enumerate(campaign["stages"]):
            if result.get(stage["agent"]):
                stage["status"] = "completed"
                stage["output"] = result[stage["agent"]]
                stage["completed_at"] = datetime.now()
        
        campaign["final_content"] = result.get("final_content")
        campaign["status"] = "completed"
        
    except Exception as e:
        campaign["status"] = "failed"
        campaign["error"] = str(e)
    
    campaign["updated_at"] = datetime.now()
    
    return CampaignResponse(**campaign)


@router.get("/campaigns/{campaign_id}/stages/{stage}")
async def get_stage_output(campaign_id: str, stage: str):
    """Get output from a specific stage."""
    if campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    campaign = campaigns_db[campaign_id]
    stage_data = next((s for s in campaign["stages"] if s["agent"] == stage), None)
    
    if not stage_data:
        raise HTTPException(status_code=404, detail="Stage not found")
    
    return stage_data


@router.delete("/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str):
    """Delete a campaign."""
    if campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    del campaigns_db[campaign_id]
    return {"message": "Campaign deleted"}
