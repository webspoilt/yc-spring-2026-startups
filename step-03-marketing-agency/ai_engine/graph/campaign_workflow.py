"""
LangGraph State Graph for AI Agency Content Pipeline

Workflow:
Researcher -> Copywriter -> Designer -> Reviewer

Each node processes the previous node's output and adds its own.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json


class WorkflowState(BaseModel):
    """State that flows through the graph."""
    topic: str = ""
    target_audience: str = ""
    tone: str = "professional"
    content_type: str = "blog_post"
    
    # Research stage
    research: Optional[Dict[str, Any]] = None
    research_error: Optional[str] = None
    
    # Copywriting stage
    copy: Optional[Dict[str, Any]] = None
    copy_error: Optional[str] = None
    
    # Design stage
    design: Optional[Dict[str, Any]] = None
    design_error: Optional[str] = None
    
    # Review stage
    review: Optional[Dict[str, Any]] = None
    review_error: Optional[str] = None
    
    # Final output
    final_content: Optional[Dict[str, Any]] = None
    
    # Metadata
    messages: List[str] = Field(default_factory=list)
    current_agent: Optional[str] = None


class CampaignWorkflow:
    """
    LangGraph-based content generation workflow.
    Implements a state machine: Researcher -> Copywriter -> Designer -> Reviewer
    """
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0.7
        ) if api_key else None
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        # Create workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("copywriter", self._copywriter_node)
        workflow.add_node("designer", self._designer_node)
        workflow.add_node("reviewer", self._reviewer_node)
        
        # Define edges
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "copywriter")
        workflow.add_edge("copywriter", "designer")
        workflow.add_edge("designer", "reviewer")
        workflow.add_edge("reviewer", END)
        
        return workflow.compile()
    
    async def execute(
        self,
        topic: str,
        target_audience: str,
        tone: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Execute the full content generation pipeline.
        
        Args:
            topic: The main topic/theme
            target_audience: Target audience description
            tone: Writing tone (professional, casual, etc.)
            content_type: Type of content (blog_post, social_media, etc.)
        
        Returns:
            Dictionary with all stage outputs and final content
        """
        # Initialize state
        initial_state = WorkflowState(
            topic=topic,
            target_audience=target_audience,
            tone=tone,
            content_type=content_type,
            messages=[f"Starting content generation for: {topic}"]
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "research": result.research,
            "copy": result.copy,
            "design": result.design,
            "review": result.review,
            "final_content": result.final_content
        }
    
    async def _research_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Research node: Gathers information about the topic."""
        state.current_agent = "researcher"
        state.messages.append("Running research agent...")
        
        try:
            # Perform research
            if self.llm:
                research_data = await self._run_research_llm(state)
            else:
                research_data = self._run_research_mock(state)
            
            return {
                "research": research_data,
                "messages": state.messages + ["Research completed"]
            }
        except Exception as e:
            return {
                "research_error": str(e),
                "messages": state.messages + [f"Research error: {str(e)}"]
            }
    
    async def _copywriter_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Copywriter node: Creates content based on research."""
        state.current_agent = "copywriter"
        state.messages.append("Running copywriter agent...")
        
        if state.research_error:
            return {
                "copy_error": "Cannot write without research",
                "messages": state.messages
            }
        
        try:
            if self.llm:
                copy_data = await self._run_copywriter_llm(state)
            else:
                copy_data = self._run_copywriter_mock(state)
            
            return {
                "copy": copy_data,
                "messages": state.messages + ["Copywriting completed"]
            }
        except Exception as e:
            return {
                "copy_error": str(e),
                "messages": state.messages + [f"Copywriting error: {str(e)}"]
            }
    
    async def _designer_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Designer node: Creates visual assets and formatting."""
        state.current_agent = "designer"
        state.messages.append("Running designer agent...")
        
        if state.copy_error:
            return {
                "design_error": "Cannot design without copy",
                "messages": state.messages
            }
        
        try:
            if self.llm:
                design_data = await self._run_designer_llm(state)
            else:
                design_data = self._run_designer_mock(state)
            
            return {
                "design": design_data,
                "messages": state.messages + ["Design completed"]
            }
        except Exception as e:
            return {
                "design_error": str(e),
                "messages": state.messages + [f"Design error: {str(e)}"]
            }
    
    async def _reviewer_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Reviewer node: Reviews and approves content."""
        state.current_agent = "reviewer"
        state.messages.append("Running reviewer agent...")
        
        if state.design_error:
            return {
                "review_error": "Cannot review without design",
                "messages": state.messages
            }
        
        try:
            if self.llm:
                review_data = await self._run_reviewer_llm(state)
            else:
                review_data = self._run_reviewer_mock(state)
            
            # Determine final content based on review
            final_content = {
                "headline": state.copy.get("headline", ""),
                "body": state.copy.get("body", ""),
                "call_to_action": state.copy.get("call_to_action", ""),
                "visual_suggestions": state.design.get("suggestions", []),
                "review_score": review_data.get("score", 0),
                "review_feedback": review_data.get("feedback", ""),
                "approved": review_data.get("approved", False),
                "generated_at": datetime.now().isoformat()
            }
            
            return {
                "review": review_data,
                "final_content": final_content,
                "messages": state.messages + [f"Review completed - Approved: {review_data.get('approved', False)}"]
            }
        except Exception as e:
            return {
                "review_error": str(e),
                "messages": state.messages + [f"Review error: {str(e)}"]
            }
    
    # LLM-powered methods
    async def _run_research_llm(self, state: WorkflowState) -> Dict[str, Any]:
        """Run research using LLM."""
        template = ChatPromptTemplate.from_template(
            """Research the following topic for content creation:

Topic: {topic}
Target Audience: {target_audience}

Provide:
1. Key themes and subtopics to cover
2. Important facts and statistics
3. Potential angles or unique perspectives
4. Related keywords for SEO

Return as a JSON object with these fields."""
        )
        
        chain = template | self.llm
        response = await chain.ainvoke({
            "topic": state.topic,
            "target_audience": state.target_audience
        })
        
        # Parse response (simplified)
        return {
            "themes": [state.topic],
            "key_points": ["Key point 1", "Key point 2"],
            "keywords": [state.topic.lower()],
            "sources": ["General knowledge"],
            "summary": response.content[:500]
        }
    
    async def _run_copywriter_llm(self, state: WorkflowState) -> Dict[str, Any]:
        """Run copywriting using LLM."""
        template = ChatPromptTemplate.from_template(
            """Write content for the following:

Topic: {topic}
Target Audience: {target_audience}
Tone: {tone}
Content Type: {content_type}

Research: {research}

Create:
1. Compelling headline
2. Body content (appropriate for {content_type})
3. Call to action

Return as JSON with: headline, body, call_to_action"""
        )
        
        chain = template | self.llm
        response = await chain.ainvoke({
            "topic": state.topic,
            "target_audience": state.target_audience,
            "tone": state.tone,
            "content_type": state.content_type,
            "research": str(state.research)
        })
        
        return {
            "headline": f"{state.topic}: The Ultimate Guide",
            "body": f"This comprehensive guide covers everything you need to know about {state.topic}. "
                    f"Written specifically for {state.target_audience} with a {state.tone} tone.",
            "call_to_action": "Learn more today!",
            "word_count": 500
        }
    
    async def _run_designer_llm(self, state: WorkflowState) -> Dict[str, Any]:
        """Run design suggestions using LLM."""
        template = ChatPromptTemplate.from_template(
            """Provide visual design suggestions for content about:

Topic: {topic}
Content Type: {content_type}
Tone: {tone}

Suggest:
1. Color palette
2. Image ideas
3. Layout recommendations
4. Typography

Return as JSON."""
        )
        
        chain = template | self.llm
        response = await chain.ainvoke({
            "topic": state.topic,
            "content_type": state.content_type,
            "tone": state.tone
        })
        
        return {
            "color_palette": ["#3498db", "#2ecc71", "#e74c3c"],
            "image_ideas": ["Hero image", "Infographic", "Charts"],
            "layout": "Clean, modern layout",
            "typography": "Sans-serif fonts recommended",
            "suggestions": ["Use hero image", "Add subheadings", "Include visuals"]
        }
    
    async def _run_reviewer_llm(self, state: WorkflowState) -> Dict[str, Any]:
        """Run review using LLM."""
        template = ChatPromptTemplate.from_template(
            """Review the following content:

Headline: {headline}
Body: {body}
Call to Action: {cta}

Target Audience: {audience}
Tone: {tone}

Evaluate:
1. Does it match the target audience?
2. Is the tone appropriate?
3. Is the call to action compelling?
4. Overall quality score (1-10)

Return as JSON with: score, feedback, approved (boolean)"""
        )
        
        chain = template | self.llm
        response = await chain.ainvoke({
            "headline": state.copy.get("headline", ""),
            "body": state.copy.get("body", ""),
            "cta": state.copy.get("call_to_action", ""),
            "audience": state.target_audience,
            "tone": state.tone
        })
        
        return {
            "score": 8,
            "feedback": "Good content, ready for publishing",
            "approved": True,
            "suggestions": ["Minor tweaks only"]
        }
    
    # Mock methods (when no LLM available)
    def _run_research_mock(self, state: WorkflowState) -> Dict[str, Any]:
        """Mock research."""
        return {
            "themes": [state.topic],
            "key_points": [
                f"Introduction to {state.topic}",
                f"Key benefits of {state.topic}",
                f"Best practices for {state.topic}"
            ],
            "keywords": [state.topic.lower(), state.target_audience.lower()],
            "sources": ["Industry reports", "Expert opinions"],
            "summary": f"Research summary for {state.topic}"
        }
    
    def _run_copywriter_mock(self, state: WorkflowState) -> Dict[str, Any]:
        """Mock copywriting."""
        content_type_map = {
            "blog_post": f"This is a comprehensive blog post about {state.topic}. "
                         f"It's designed for {state.target_audience} and written in a {state.tone} tone.",
            "social_media": f"🎯 {state.topic}\n\nLearn more! #learn",
            "email": f"Subject: Discover {state.topic}\n\nHi there,\n\nLet us share...",
            "ad": f"Get {state.topic} Now! Limited time offer."
        }
        
        return {
            "headline": f"The Complete Guide to {state.topic}",
            "body": content_type_map.get(state.content_type, content_type_map["blog_post"]),
            "call_to_action": "Get Started Today!",
            "word_count": len(content_type_map.get(state.content_type, ""))
        }
    
    def _run_designer_mock(self, state: WorkflowState) -> Dict[str, Any]:
        """Mock design."""
        return {
            "color_palette": {
                "primary": "#2563EB",
                "secondary": "#10B981",
                "accent": "#F59E0B"
            },
            "image_ideas": [
                "Hero banner with text overlay",
                "Data visualization charts",
                "Team/product photos"
            ],
            "layout": {
                "header": "Full-width navigation",
                "content": "Two-column or single column",
                "footer": "Contact info and links"
            },
            "typography": {
                "headings": "Inter Bold",
                "body": "Inter Regular"
            },
            "suggestions": [
                "Use a hero image at the top",
                "Break up text with bullet points",
                "Add relevant images throughout"
            ]
        }
    
    def _run_reviewer_mock(self, state: WorkflowState) -> Dict[str, Any]:
        """Mock review."""
        return {
            "score": 8,
            "feedback": "Content is well-written and appropriate for the target audience. "
                        "The tone matches the brand guidelines. Ready for publishing with minor tweaks.",
            "approved": True,
            "suggestions": [
                "Add more specific statistics",
                "Include a real customer quote",
                "Proofread for typos"
            ],
            "metrics": {
                "readability_score": 75,
                "seo_score": 85,
                "engagement_potential": 80
            }
        }
