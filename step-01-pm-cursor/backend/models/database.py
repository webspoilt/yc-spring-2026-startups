from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class Priority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, enum.Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user_stories = relationship("UserStory", back_populates="project")
    specifications = relationship("Specification", back_populates="project")


class UserStory(Base):
    __tablename__ = "user_stories"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    as_a = Column(String(255))
    i_want = Column(String(255))
    so_that = Column(String(255))
    priority = Column(SQLEnum(Priority), default=Priority.MEDIUM)
    status = Column(SQLEnum(Status), default=Status.DRAFT)
    acceptance_criteria = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    project = relationship("Project", back_populates="user_stories")


class Specification(Base):
    __tablename__ = "specifications"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)  # Tiptap JSON content
    version = Column(Integer, default=1)
    status = Column(SQLEnum(Status), default=Status.DRAFT)
    generated_by_ai = Column(String(50))  # 'langchain', 'openai', etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    project = relationship("Project", back_populates="specifications")
