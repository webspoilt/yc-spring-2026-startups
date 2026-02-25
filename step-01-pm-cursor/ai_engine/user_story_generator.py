import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class UserStoryGenerator:
    """
    AI-powered user story generator using LangChain.
    Generates well-structured user stories from feature descriptions.
    """
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0.7
        ) if api_key else None
    
    def generate(
        self, 
        project_name: str, 
        feature_description: str, 
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate user stories from a feature description.
        
        Args:
            project_name: Name of the project
            feature_description: Description of the feature
            count: Number of user stories to generate
        
        Returns:
            List of user story dictionaries
        """
        if not self.llm:
            return self._generate_mock(feature_description, count)
        
        template = ChatPromptTemplate.from_template(
            """You are a Senior Product Manager. Generate {count} detailed user stories for a feature.

Project: {project_name}
Feature: {feature_description}

For each user story, provide:
- title: A concise title for the story
- description: Detailed description of what the user wants
- as_a: The user role (e.g., "registered user", "admin", "guest")
- i_want: The action the user wants to perform
- so_that: The benefit/outcome for the user
- priority: One of "high", "medium", or "low"
- acceptance_criteria: 3-5 bullet points of measurable acceptance criteria

Return your response as a JSON array of user story objects. Each story should follow this format:
[
    {{
        "title": "Story Title",
        "description": "Detailed description...",
        "as_a": "user role",
        "i_want": "desired action",
        "so_that": "beneficial outcome",
        "priority": "high/medium/low",
        "acceptance_criteria": "1. First criterion\\n2. Second criterion\\n3. Third criterion"
    }}
]

Generate exactly {count} user stories."""
        )
        
        chain = template | self.llm
        
        response = chain.invoke({
            "project_name": project_name,
            "feature_description": feature_description,
            "count": count
        })
        
        return self._parse_response(response.content, count)
    
    def _generate_mock(self, feature_description: str, count: int) -> List[Dict[str, Any]]:
        """Fallback mock generation when no API key is available."""
        return [
            {
                "title": "User Authentication",
                "description": f"As a user, I want to authenticate so that I can access the feature: {feature_description[:50]}",
                "as_a": "registered user",
                "i_want": "to log in with my credentials",
                "so_that": "I can access my personalized content",
                "priority": "high",
                "acceptance_criteria": "1. Login form accepts email and password\n2. Invalid credentials show error\n3. Successful login redirects to dashboard"
            },
            {
                "title": "Data Management",
                "description": "As a user, I want to manage my data",
                "as_a": "registered user",
                "i_want": "to create, read, update, and delete my data",
                "so_that": "I have full control over my information",
                "priority": "medium",
                "acceptance_criteria": "1. CRUD operations work correctly\n2. Changes are persisted\n3. User sees confirmation messages"
            }
        ][:count]
    
    def _parse_response(self, response: str, expected_count: int) -> List[Dict[str, Any]]:
        """Parse the LLM response into a list of user stories."""
        import json
        
        try:
            # Try to extract JSON array from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                stories = json.loads(json_str)
                if isinstance(stories, list):
                    return stories[:expected_count]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback to mock
        return self._generate_mock("feature", expected_count)
    
    def generate_from_acceptance_criteria(
        self,
        acceptance_criteria: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate testable acceptance criteria into individual user stories.
        
        Args:
            acceptance_criteria: The acceptance criteria text
            count: Maximum number of stories
        
        Returns:
            List of refined acceptance criteria as stories
        """
        if not self.llm:
            return [{"title": f"AC-{i+1}", "description": line.strip()} 
                    for i, line in enumerate(acceptance_criteria.split('\n')) 
                    if line.strip()][:count]
        
        template = ChatPromptTemplate.from_template(
            """Break down the following acceptance criteria into {count} individual, testable scenarios.

Acceptance Criteria:
{acceptance_criteria}

Return as a JSON array of test scenarios:
[
    {{"title": "Scenario 1", "description": "Testable scenario description"}},
    ...
]"""
        )
        
        chain = template | self.llm
        response = chain.invoke({
            "acceptance_criteria": acceptance_criteria,
            "count": count
        })
        
        return self._parse_response(response.content, count)
