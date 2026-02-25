import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel


class SpecGenerator:
    """
    AI-powered specification generator using LangChain.
    Generates detailed, technical specifications in Tiptap JSON format.
    """
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0.7
        ) if api_key else None
    
    def generate(self, project_name: str, prompt: str, style: str = "technical") -> Dict[str, Any]:
        """
        Generate a specification document.
        
        Args:
            project_name: Name of the project
            prompt: User's requirement prompt
            style: Style of specification (technical, user-facing, executive)
        
        Returns:
            Dictionary with title, Tiptap content, and user stories
        """
        if not self.llm:
            return self._generate_mock(project_name, prompt, style)
        
        style_templates = {
            "technical": self._technical_prompt,
            "user-facing": self._user_facing_prompt,
            "executive": self._executive_prompt
        }
        
        template = style_templates.get(style, self._technical_prompt)
        
        chain = template | self.llm
        
        response = chain.invoke({
            "project_name": project_name,
            "prompt": prompt
        })
        
        return self._parse_response(response.content, prompt)
    
    def _generate_mock(self, project_name: str, prompt: str, style: str) -> Dict[str, Any]:
        """Fallback mock generation when no API key is available."""
        return {
            "title": f"Specification for {project_name}",
            "content": self._create_tiptap_content(prompt),
            "user_stories": [
                {
                    "title": "User Authentication",
                    "description": "As a user, I want to authenticate securely so that I can access my account.",
                    "as_a": "registered user",
                    "i_want": "secure login functionality",
                    "so_that": "I can access my personalized dashboard",
                    "priority": "high",
                    "acceptance_criteria": "1. User can login with email/password\n2. Session expires after 30 minutes\n3. Failed attempts show error message"
                }
            ]
        }
    
    def _technical_prompt(self):
        return ChatPromptTemplate.from_template(
            """You are a Senior Technical Product Manager. Generate a detailed technical specification for a software project.

Project Name: {project_name}
User Request: {prompt}

Generate a comprehensive technical specification that includes:
1. A clear title
2. Technical requirements (API endpoints, data models, integrations)
3. User stories derived from the requirements
4. Acceptance criteria for each story

Return your response as a JSON object with this exact structure:
{{
    "title": "Specification Title",
    "content": {{
        "type": "doc",
        "content": [
            {{"type": "heading", "attrs": {{"level": 1}}, "content": [{{"type": "text", "text": "Title"}}]}},
            {{"type": "heading", "attrs": {{"level": 2}}, "content": [{{"type": "text", "text": "Overview"}}]}},
            {{"type": "paragraph", "content": [{{"type": "text", "text": "Description"}}]}}
        ]
    }},
    "user_stories": [
        {{"title": "...", "description": "...", "as_a": "...", "i_want": "...", "so_that": "...", "priority": "high/medium/low", "acceptance_criteria": "..."}}
    ]
}}

Ensure the content is in Tiptap JSON format for rich text editing."""
        )
    
    def _user_facing_prompt(self):
        return ChatPromptTemplate.from_template(
            """You are a Product Manager focused on user experience. Generate a user-friendly specification for a software project.

Project Name: {project_name}
User Request: {prompt}

Generate a specification that focuses on:
1. User workflows and journey
2. UI/UX requirements
3. User stories from the end-user perspective
4. Clear acceptance criteria

Return your response as a JSON object with this exact structure:
{{
    "title": "Specification Title",
    "content": {{...Tiptap JSON format...}},
    "user_stories": [...]
}}"""
        )
    
    def _executive_prompt(self):
        return ChatPromptTemplate.from_template(
            """You are an Enterprise Architect. Generate an executive-level specification for a software project.

Project Name: {project_name}
User Request: {prompt}

Generate a specification that focuses on:
1. Business value and ROI
2. High-level requirements
3. Key success metrics
4. Strategic alignment

Return your response as a JSON object with this exact structure:
{{
    "title": "Specification Title",
    "content": {{...Tiptap JSON format...}},
    "user_stories": [...]
}}"""
        )
    
    def _parse_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        import json
        
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback to default structure
        return {
            "title": f"Specification - {original_prompt[:50]}",
            "content": self._create_tiptap_content(original_prompt),
            "user_stories": []
        }
    
    def _create_tiptap_content(self, prompt: str) -> Dict[str, Any]:
        """Create Tiptap JSON content structure."""
        return {
            "type": "doc",
            "content": [
                {
                    "type": "heading",
                    "attrs": {"level": 1},
                    "content": [{"type": "text", "text": "Overview"}]
                },
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": prompt}]
                },
                {
                    "type": "heading",
                    "attrs": {"level": 2},
                    "content": [{"type": "text", "text": "Technical Requirements"}]
                },
                {
                    "type": "bulletList",
                    "content": [
                        {
                            "type": "listItem",
                            "content": [{
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "API endpoints to be defined"}]
                            }]
                        },
                        {
                            "type": "listItem",
                            "content": [{
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "Database schema to be designed"}]
                            }]
                        }
                    ]
                }
            ]
        }
