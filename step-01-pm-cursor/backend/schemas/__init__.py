from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# Project Schemas
class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectResponse(ProjectBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# User Story Schemas
class UserStoryBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str
    as_a: Optional[str] = None
    i_want: Optional[str] = None
    so_that: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    status: Status = Status.DRAFT
    acceptance_criteria: Optional[str] = None


class UserStoryCreate(UserStoryBase):
    project_id: int


class UserStoryUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    as_a: Optional[str] = None
    i_want: Optional[str] = None
    so_that: Optional[str] = None
    priority: Optional[Priority] = None
    status: Optional[Status] = None
    acceptance_criteria: Optional[str] = None


class UserStoryResponse(UserStoryBase):
    id: int
    project_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Specification Schemas
class SpecificationBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str  # Tiptap JSON
    version: int = 1
    status: Status = Status.DRAFT
    generated_by_ai: Optional[str] = None


class SpecificationCreate(SpecificationBase):
    project_id: int


class SpecificationUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = None
    status: Optional[Status] = None
    generated_by_ai: Optional[str] = None


class SpecificationResponse(SpecificationBase):
    id: int
    project_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# AI Generation Schemas
class GenerateSpecRequest(BaseModel):
    project_id: int
    prompt: str = Field(..., min_length=10, max_length=2000)
    style: Optional[str] = "technical"  # technical, user-facing, executive


class GenerateSpecResponse(BaseModel):
    specification_id: int
    title: str
    content: str
    generated_by_ai: str
    user_stories_generated: int = 0


class GenerateUserStoriesRequest(BaseModel):
    project_id: int
    feature_description: str = Field(..., min_length=20)
    count: int = Field(default=3, ge=1, le=10)


class GenerateUserStoriesResponse(BaseModel):
    user_stories: List[UserStoryResponse]
    generation_method: str
