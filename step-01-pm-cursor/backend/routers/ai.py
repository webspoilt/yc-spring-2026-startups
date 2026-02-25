from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import json

from ..database import get_db
from ..models.database import Project, UserStory, Specification, Status as SpecStatus
from ..schemas import (
    GenerateSpecRequest, GenerateSpecResponse,
    GenerateUserStoriesRequest, GenerateUserStoriesResponse,
    UserStoryResponse, Priority, Status
)
from ...ai_engine.spec_generator import SpecGenerator
from ...ai_engine.user_story_generator import UserStoryGenerator

router = APIRouter()

# Initialize AI engines
spec_generator = SpecGenerator()
user_story_generator = UserStoryGenerator()


@router.post("/generate-spec", response_model=GenerateSpecResponse)
def generate_specification(request: GenerateSpecRequest, db: Session = Depends(get_db)):
    """
    Generate a detailed specification using AI based on the user's prompt.
    Returns Tiptap-compatible JSON content.
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Generate specification using AI
    try:
        generated_spec = spec_generator.generate(
            project_name=project.name,
            prompt=request.prompt,
            style=request.style
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate specification: {str(e)}"
        )
    
    # Create specification in database
    db_spec = Specification(
        project_id=request.project_id,
        title=generated_spec["title"],
        content=json.dumps(generated_spec["content"]),
        status=SpecStatus.DRAFT,
        generated_by_ai="langchain"
    )
    db.add(db_spec)
    db.commit()
    db.refresh(db_spec)
    
    # Generate user stories from the specification
    user_stories_generated = 0
    if generated_spec.get("user_stories"):
        for story_data in generated_spec["user_stories"]:
            db_story = UserStory(
                project_id=request.project_id,
                title=story_data["title"],
                description=story_data["description"],
                as_a=story_data.get("as_a"),
                i_want=story_data.get("i_want"),
                so_that=story_data.get("so_that"),
                priority=Priority(story_data.get("priority", "medium")),
                status=Status.DRAFT,
                acceptance_criteria=story_data.get("acceptance_criteria")
            )
            db.add(db_story)
            user_stories_generated += 1
        db.commit()
    
    return GenerateSpecResponse(
        specification_id=db_spec.id,
        title=db_spec.title,
        content=db_spec.content,
        generated_by_ai=db_spec.generated_by_ai,
        user_stories_generated=user_stories_generated
    )


@router.post("/generate-user-stories", response_model=GenerateUserStoriesResponse)
def generate_user_stories(request: GenerateUserStoriesRequest, db: Session = Depends(get_db)):
    """
    Generate user stories from a feature description using AI.
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Generate user stories using AI
    try:
        stories = user_story_generator.generate(
            project_name=project.name,
            feature_description=request.feature_description,
            count=request.count
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate user stories: {str(e)}"
        )
    
    # Save user stories to database
    created_stories = []
    for story_data in stories:
        db_story = UserStory(
            project_id=request.project_id,
            title=story_data["title"],
            description=story_data["description"],
            as_a=story_data.get("as_a"),
            i_want=story_data.get("i_want"),
            so_that=story_data.get("so_that"),
            priority=Priority(story_data.get("priority", "medium")),
            status=Status.DRAFT,
            acceptance_criteria=story_data.get("acceptance_criteria")
        )
        db.add(db_story)
        db.commit()
        db.refresh(db_story)
        created_stories.append(db_story)
    
    return GenerateUserStoriesResponse(
        user_stories=[UserStoryResponse.model_validate(s) for s in created_stories],
        generation_method="langchain"
    )


@router.get("/ai/capabilities")
def get_ai_capabilities():
    """Return available AI generation capabilities."""
    return {
        "spec_generation": {
            "enabled": True,
            "styles": ["technical", "user-facing", "executive"],
            "model": "gpt-3.5-turbo"
        },
        "user_story_generation": {
            "enabled": True,
            "max_stories_per_request": 10,
            "model": "gpt-3.5-turbo"
        }
    }
