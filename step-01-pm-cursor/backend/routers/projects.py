from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db, init_db
from ..models.database import Project, UserStory, Specification
from ..schemas import (
    ProjectCreate, ProjectUpdate, ProjectResponse,
    UserStoryCreate, UserStoryUpdate, UserStoryResponse,
    SpecificationCreate, SpecificationUpdate, SpecificationResponse,
)

router = APIRouter()


@router.on_event("startup")
def startup_event():
    init_db()


# Project Endpoints
@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    db_project = Project(**project.model_dump())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


@router.get("/projects", response_model=List[ProjectResponse])
def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(Project).offset(skip).limit(limit).all()


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    update_data = project.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_project, field, value)
    
    db.commit()
    db.refresh(db_project)
    return db_project


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
    return None


# User Story Endpoints
@router.post("/projects/{project_id}/user-stories", response_model=UserStoryResponse, status_code=status.HTTP_201_CREATED)
def create_user_story(project_id: int, story: UserStoryCreate, db: Session = Depends(get_db)):
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db_story = UserStory(project_id=project_id, **story.model_dump(exclude={"project_id"}))
    db.add(db_story)
    db.commit()
    db.refresh(db_story)
    return db_story


@router.get("/projects/{project_id}/user-stories", response_model=List[UserStoryResponse])
def list_user_stories(project_id: int, db: Session = Depends(get_db)):
    stories = db.query(UserStory).filter(UserStory.project_id == project_id).all()
    return stories


@router.get("/user-stories/{story_id}", response_model=UserStoryResponse)
def get_user_story(story_id: int, db: Session = Depends(get_db)):
    story = db.query(UserStory).filter(UserStory.id == story_id).first()
    if not story:
        raise HTTPException(status_code=404, detail="User story not found")
    return story


@router.patch("/user-stories/{story_id}", response_model=UserStoryResponse)
def update_user_story(story_id: int, story: UserStoryUpdate, db: Session = Depends(get_db)):
    db_story = db.query(UserStory).filter(UserStory.id == story_id).first()
    if not db_story:
        raise HTTPException(status_code=404, detail="User story not found")
    
    update_data = story.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_story, field, value)
    
    db.commit()
    db.refresh(db_story)
    return db_story


@router.delete("/user-stories/{story_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_story(story_id: int, db: Session = Depends(get_db)):
    story = db.query(UserStory).filter(UserStory.id == story_id).first()
    if not story:
        raise HTTPException(status_code=404, detail="User story not found")
    db.delete(story)
    db.commit()
    return None


# Specification Endpoints
@router.post("/projects/{project_id}/specifications", response_model=SpecificationResponse, status_code=status.HTTP_201_CREATED)
def create_specification(project_id: int, spec: SpecificationCreate, db: Session = Depends(get_db)):
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db_spec = Specification(project_id=project_id, **spec.model_dump(exclude={"project_id"}))
    db.add(db_spec)
    db.commit()
    db.refresh(db_spec)
    return db_spec


@router.get("/projects/{project_id}/specifications", response_model=List[SpecificationResponse])
def list_specifications(project_id: int, db: Session = Depends(get_db)):
    specs = db.query(Specification).filter(Specification.project_id == project_id).all()
    return specs


@router.get("/specifications/{spec_id}", response_model=SpecificationResponse)
def get_specification(spec_id: int, db: Session = Depends(get_db)):
    spec = db.query(Specification).filter(Specification.id == spec_id).first()
    if not spec:
        raise HTTPException(status_code=404, detail="Specification not found")
    return spec


@router.patch("/specifications/{spec_id}", response_model=SpecificationResponse)
def update_specification(spec_id: int, spec: SpecificationUpdate, db: Session = Depends(get_db)):
    db_spec = db.query(Specification).filter(Specification.id == spec_id).first()
    if not db_spec:
        raise HTTPException(status_code=404, detail="Specification not found")
    
    update_data = spec.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_spec, field, value)
    
    db.commit()
    db.refresh(db_spec)
    return db_spec


@router.delete("/specifications/{spec_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_specification(spec_id: int, db: Session = Depends(get_db)):
    spec = db.query(Specification).filter(Specification.id == spec_id).first()
    if not spec:
        raise HTTPException(status_code=404, detail="Specification not found")
    db.delete(spec)
    db.commit()
    return None
