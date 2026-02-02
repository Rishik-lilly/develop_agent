"""
JIRA Dump to Code Generator
Supports both real LLM generation (Azure OpenAI) and mocked fallback
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from enum import Enum
import requests

# Suppress SSL warnings for corporate environments
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# Configuration
# ============================================================================

class GenerationMode(Enum):
    REAL = "real"      # Use actual LLM API (Azure OpenAI)
    MOCK = "mock"      # Use templates (fast, no API needed)
    AUTO = "auto"      # Try real, fallback to mock

@dataclass
class Config:
    mode: GenerationMode = GenerationMode.AUTO
    output_dir: str = "./generated_code"
    
    # Azure OpenAI Configuration
    azure_endpoint: str = "https://openai-gis-sdlc-automation-instance.openai.azure.com/"
    azure_api_key: str = "81dcede85f694a0a8a4ab2874950b6cc"
    azure_deployment: str = "gpt-4o-mini"
    azure_api_version: str = "2024-12-01-preview"
    
    stream: bool = True

# ============================================================================
# JIRA Parser
# ============================================================================

class JiraParser:
    """Parse JIRA dump and extract relevant information"""
    
    def __init__(self, dump_path: str):
        self.dump_path = dump_path
        self.data = self._load_dump()
    
    def _load_dump(self) -> Dict[str, Any]:
        with open(self.dump_path, 'r') as f:
            return json.load(f)
    
    def get_issues(self) -> List[Dict[str, Any]]:
        return self.data.get("issues", [])
    
    def get_issue_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        for issue in self.get_issues():
            if issue.get("key") == key:
                return issue
        return None
    
    def detect_code_type(self, issue: Dict[str, Any]) -> str:
        """Detect if issue is for backend API or frontend component"""
        labels = issue.get("labels", [])
        tech_stack = issue.get("tech_stack", [])
        summary = issue.get("summary", "").lower()
        
        frontend_indicators = ["react", "frontend", "component", "ui", "widget"]
        backend_indicators = ["api", "backend", "service", "endpoint", "python", "fastapi"]
        
        frontend_score = sum(1 for ind in frontend_indicators 
                           if any(ind in str(v).lower() for v in labels + tech_stack + [summary]))
        backend_score = sum(1 for ind in backend_indicators 
                          if any(ind in str(v).lower() for v in labels + tech_stack + [summary]))
        
        return "frontend" if frontend_score > backend_score else "backend"

# ============================================================================
# Prompt Builder
# ============================================================================

class PromptBuilder:
    """Build prompts for code generation"""
    
    @staticmethod
    def build_backend_prompt(issue: Dict[str, Any]) -> str:
        ac = "\n".join("  - " + c for c in issue.get("acceptance_criteria", []))
        tech = ", ".join(issue.get("tech_stack", ["Python", "FastAPI"]))
        
        return """Generate a complete, production-ready Python FastAPI implementation for the following JIRA ticket.

## JIRA Ticket: {key}
**Summary:** {summary}

**Description:** 
{description}

**Acceptance Criteria:**
{ac}

**Tech Stack:** {tech}

## Requirements:
1. Use FastAPI with proper type hints and Pydantic models
2. Include all necessary imports
3. Add docstrings and comments
4. Implement proper error handling with HTTPException
5. Use async/await where appropriate
6. Include input validation
7. Follow REST best practices

Generate ONLY the Python code, no explanations. Start with imports.""".format(
            key=issue.get('key'),
            summary=issue.get('summary'),
            description=issue.get('description'),
            ac=ac,
            tech=tech
        )

    @staticmethod
    def build_frontend_prompt(issue: Dict[str, Any]) -> str:
        ac = "\n".join("  - " + c for c in issue.get("acceptance_criteria", []))
        tech = ", ".join(issue.get("tech_stack", ["React", "TypeScript"]))
        
        return """Generate a complete, production-ready React TypeScript component for the following JIRA ticket.

## JIRA Ticket: {key}
**Summary:** {summary}

**Description:**
{description}

**Acceptance Criteria:**
{ac}

**Tech Stack:** {tech}

## Requirements:
1. Use React functional components with TypeScript
2. Use hooks (useState, useEffect, etc.) appropriately
3. Include proper TypeScript interfaces/types
4. Use TailwindCSS for styling
5. Handle loading and error states
6. Make the component reusable with props
7. Add JSDoc comments

Generate ONLY the TSX code, no explanations. Start with imports.""".format(
            key=issue.get('key'),
            summary=issue.get('summary'),
            description=issue.get('description'),
            ac=ac,
            tech=tech
        )

# ============================================================================
# Real LLM Generator (Azure OpenAI)
# ============================================================================

class RealCodeGenerator:
    """Generate code using Azure OpenAI API"""
    
    def __init__(self, config: Config):
        self.config = config
        self.endpoint = config.azure_endpoint
        self.api_key = config.azure_api_key
        self.deployment = config.azure_deployment
        self.api_version = config.azure_api_version
    
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    def generate(self, prompt: str) -> Generator[str, None, None]:
        """Generate code with streaming response from Azure OpenAI"""
        if not self.is_available():
            raise RuntimeError("Azure OpenAI API key not configured")
        
        url = f"{self.endpoint}openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert software engineer. Generate clean, production-ready code. Output ONLY code, no explanations or markdown code blocks."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.7,
            "stream": self.config.stream
        }
        
        try:
            if self.config.stream:
                yield from self._stream_response(url, headers, payload)
            else:
                yield self._batch_response(url, headers, payload)
        except Exception as e:
            raise RuntimeError("Azure OpenAI API call failed: " + str(e))
    
    def _stream_response(self, url: str, headers: dict, payload: dict) -> Generator[str, None, None]:
        """Handle streaming response from Azure OpenAI"""
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            verify=False,  # For corporate SSL inspection
            timeout=60
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    
    def _batch_response(self, url: str, headers: dict, payload: dict) -> str:
        """Handle non-streaming response from Azure OpenAI"""
        payload["stream"] = False
        response = requests.post(
            url, 
            headers=headers, 
            json=payload,
            verify=False,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

# ============================================================================
# Mock Code Generator (Template-based)
# ============================================================================

class MockCodeGenerator:
    """Generate code using templates - no API needed"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate(self, issue: Dict[str, Any], code_type: str) -> Generator[str, None, None]:
        """Generate mocked code with simulated streaming"""
        if code_type == "frontend":
            code = self._generate_frontend(issue)
        else:
            code = self._generate_backend(issue)
        
        # Simulate streaming by yielding chunks
        chunk_size = 50
        for i in range(0, len(code), chunk_size):
            yield code[i:i + chunk_size]
            time.sleep(0.02)  # Simulate network delay
    
    def _generate_backend(self, issue: Dict[str, Any]) -> str:
        """Generate backend Python/FastAPI code from template"""
        key = issue.get("key", "DEMO-000")
        summary = issue.get("summary", "API Endpoint")
        description = issue.get("description", "")
        
        # Extract resource name from summary
        resource = self._extract_resource_name(summary)
        resource_lower = resource.lower()
        
        code = '''"""
{summary}
JIRA: {key}

{description}
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/{resource_lower}", tags=["{resource}"])

# ============================================================================
# Models
# ============================================================================

class {resource}Base(BaseModel):
    """Base model for {resource}"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    class Config:
        json_schema_extra = {{
            "example": {{
                "name": "Example {resource}",
                "description": "A sample description"
            }}
        }}

class {resource}Create({resource}Base):
    """Model for creating a new {resource}"""
    pass

class {resource}Update(BaseModel):
    """Model for updating an existing {resource}"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)

class {resource}Response({resource}Base):
    """Response model for {resource}"""
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class {resource}ListResponse(BaseModel):
    """Paginated list response"""
    items: List[{resource}Response]
    total: int
    page: int
    limit: int
    has_more: bool

# ============================================================================
# Dependencies
# ============================================================================

async def get_current_user():
    """Dependency to get current authenticated user"""
    # TODO: Implement actual authentication
    return {{"id": "user_123", "email": "user@example.com"}}

# ============================================================================
# Endpoints
# ============================================================================

@router.get("/", response_model={resource}ListResponse)
async def list_{resource_lower}s(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query"),
    current_user: dict = Depends(get_current_user)
):
    """
    List all {resource_lower}s with pagination and optional search.
    
    - **page**: Page number (starts from 1)
    - **limit**: Number of items per page (max 100)
    - **search**: Optional search query
    """
    logger.info(f"Listing {resource_lower}s for user {{current_user['id']}}")
    
    try:
        # TODO: Implement actual database query
        items = []
        total = 0
        
        return {resource}ListResponse(
            items=items,
            total=total,
            page=page,
            limit=limit,
            has_more=(page * limit) < total
        )
    except Exception as e:
        logger.error(f"Error listing {resource_lower}s: {{e}}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{{item_id}}", response_model={resource}Response)
async def get_{resource_lower}(
    item_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get a specific {resource_lower} by ID.
    
    - **item_id**: The unique identifier of the {resource_lower}
    """
    logger.info(f"Getting {resource_lower} {{item_id}} for user {{current_user['id']}}")
    
    # TODO: Implement actual database lookup
    raise HTTPException(status_code=404, detail="{resource} not found")

@router.post("/", response_model={resource}Response, status_code=201)
async def create_{resource_lower}(
    data: {resource}Create,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new {resource_lower}.
    
    - **data**: The {resource_lower} data to create
    """
    logger.info(f"Creating {resource_lower} for user {{current_user['id']}}")
    
    try:
        # TODO: Implement actual creation logic
        now = datetime.utcnow()
        
        return {resource}Response(
            id="new_id",
            name=data.name,
            description=data.description,
            created_at=now,
            updated_at=now
        )
    except Exception as e:
        logger.error(f"Error creating {resource_lower}: {{e}}")
        raise HTTPException(status_code=500, detail="Failed to create {resource_lower}")

@router.put("/{{item_id}}", response_model={resource}Response)
async def update_{resource_lower}(
    item_id: str,
    data: {resource}Update,
    current_user: dict = Depends(get_current_user)
):
    """
    Update an existing {resource_lower}.
    
    - **item_id**: The unique identifier of the {resource_lower}
    - **data**: The fields to update
    """
    logger.info(f"Updating {resource_lower} {{item_id}} for user {{current_user['id']}}")
    
    # TODO: Implement actual update logic
    raise HTTPException(status_code=404, detail="{resource} not found")

@router.delete("/{{item_id}}", status_code=204)
async def delete_{resource_lower}(
    item_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a {resource_lower}.
    
    - **item_id**: The unique identifier of the {resource_lower} to delete
    """
    logger.info(f"Deleting {resource_lower} {{item_id}} for user {{current_user['id']}}")
    
    # TODO: Implement actual deletion logic
    raise HTTPException(status_code=404, detail="{resource} not found")
'''.format(
            summary=summary,
            key=key,
            description=description,
            resource=resource,
            resource_lower=resource_lower
        )
        return code
    
    def _generate_frontend(self, issue: Dict[str, Any]) -> str:
        """Generate frontend React/TypeScript code from template"""
        key = issue.get("key", "DEMO-000")
        summary = issue.get("summary", "Component")
        description = issue.get("description", "")
        
        # Extract component name
        component = self._extract_component_name(summary)
        
        code = '''/**
 * {summary}
 * JIRA: {key}
 * 
 * {description}
 */

import React, {{ useState, useEffect, useCallback }} from 'react';

// ============================================================================
// Types
// ============================================================================

interface {component}Data {{
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
}}

interface {component}Props {{
  /** Initial data to display */
  initialData?: {component}Data;
  /** Callback when data is saved */
  onSave?: (data: {component}Data) => void;
  /** Callback when component is closed */
  onClose?: () => void;
  /** Whether the component is in read-only mode */
  readOnly?: boolean;
  /** Custom CSS class name */
  className?: string;
}}

interface FormState {{
  name: string;
  description: string;
}}

interface ValidationErrors {{
  name?: string;
  description?: string;
}}

// ============================================================================
// Component
// ============================================================================

export const {component}: React.FC<{component}Props> = ({{
  initialData,
  onSave,
  onClose,
  readOnly = false,
  className = '',
}}) => {{
  // State
  const [isLoading, setIsLoading] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<FormState>({{
    name: initialData?.name || '',
    description: initialData?.description || '',
  }});
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({{}});

  // Effects
  useEffect(() => {{
    if (initialData) {{
      setFormData({{
        name: initialData.name,
        description: initialData.description || '',
      }});
    }}
  }}, [initialData]);

  // Validation
  const validate = useCallback((): boolean => {{
    const errors: ValidationErrors = {{}};
    
    if (!formData.name.trim()) {{
      errors.name = 'Name is required';
    }} else if (formData.name.length < 2) {{
      errors.name = 'Name must be at least 2 characters';
    }}
    
    if (formData.description && formData.description.length > 500) {{
      errors.description = 'Description must be less than 500 characters';
    }}
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  }}, [formData]);

  // Handlers
  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {{
    const {{ name, value }} = e.target;
    setFormData(prev => ({{ ...prev, [name]: value }}));
    
    if (validationErrors[name as keyof ValidationErrors]) {{
      setValidationErrors(prev => ({{ ...prev, [name]: undefined }}));
    }}
  }};

  const handleSubmit = async (e: React.FormEvent) => {{
    e.preventDefault();
    
    if (!validate()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {{
      const savedData: {component}Data = {{
        id: initialData?.id || 'new_id',
        name: formData.name,
        description: formData.description,
        createdAt: initialData?.createdAt || new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }};
      
      onSave?.(savedData);
      setIsEditing(false);
    }} catch (err) {{
      setError(err instanceof Error ? err.message : 'An error occurred');
    }} finally {{
      setIsLoading(false);
    }}
  }};

  const handleCancel = () => {{
    setFormData({{
      name: initialData?.name || '',
      description: initialData?.description || '',
    }});
    setValidationErrors({{}});
    setIsEditing(false);
  }};

  // Render loading state
  if (isLoading && !isEditing) {{
    return (
      <div className={{`${{className}} animate-pulse`}}>
        <div className="h-8 bg-gray-200 rounded w-3/4 mb-4"></div>
        <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-5/6"></div>
      </div>
    );
  }}

  return (
    <div className={{`${{className}} bg-white rounded-lg shadow-md p-6`}}>
      {{/* Header */}}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-800">
          {{initialData ? 'Edit' : 'Create'}} {component_display}
        </h2>
        {{onClose && (
          <button
            onClick={{onClose}}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={{2}} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}}
      </div>

      {{/* Error Message */}}
      {{error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-600 text-sm">{{error}}</p>
        </div>
      )}}

      {{/* Form */}}
      <form onSubmit={{handleSubmit}} className="space-y-4">
        {{/* Name Field */}}
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
            Name <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={{formData.name}}
            onChange={{handleInputChange}}
            disabled={{readOnly || isLoading}}
            className={{`
              w-full px-3 py-2 border rounded-md shadow-sm
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
              disabled:bg-gray-100 disabled:cursor-not-allowed
              ${{validationErrors.name ? 'border-red-500' : 'border-gray-300'}}
            `}}
            placeholder="Enter name"
          />
          {{validationErrors.name && (
            <p className="mt-1 text-sm text-red-500">{{validationErrors.name}}</p>
          )}}
        </div>

        {{/* Description Field */}}
        <div>
          <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
            Description
          </label>
          <textarea
            id="description"
            name="description"
            value={{formData.description}}
            onChange={{handleInputChange}}
            disabled={{readOnly || isLoading}}
            rows={{4}}
            className={{`
              w-full px-3 py-2 border rounded-md shadow-sm
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
              disabled:bg-gray-100 disabled:cursor-not-allowed
              ${{validationErrors.description ? 'border-red-500' : 'border-gray-300'}}
            `}}
            placeholder="Enter description (optional)"
          />
          {{validationErrors.description && (
            <p className="mt-1 text-sm text-red-500">{{validationErrors.description}}</p>
          )}}
          <p className="mt-1 text-xs text-gray-500">
            {{formData.description.length}}/500 characters
          </p>
        </div>

        {{/* Actions */}}
        {{!readOnly && (
          <div className="flex justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={{handleCancel}}
              disabled={{isLoading}}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={{isLoading}}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 flex items-center"
            >
              {{isLoading && (
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              )}}
              {{isLoading ? 'Saving...' : 'Save'}}
            </button>
          </div>
        )}}
      </form>
    </div>
  );
}};

export default {component};
'''.format(
            summary=summary,
            key=key,
            description=description,
            component=component,
            component_display=self._make_display_name(component)
        )
        return code
    
    def _extract_resource_name(self, summary: str) -> str:
        """Extract a resource name from summary for backend APIs"""
        words = summary.replace("Create", "").replace("Build", "").replace("Implement", "")
        words = words.replace("API", "").replace("Service", "").replace("Endpoint", "")
        words = ''.join(c for c in words if c.isalnum() or c.isspace())
        parts = [w.capitalize() for w in words.split() if len(w) > 2]
        return parts[0] if parts else "Resource"
    
    def _extract_component_name(self, summary: str) -> str:
        """Extract a component name from summary for frontend"""
        words = summary.replace("Create", "").replace("Build", "").replace("Implement", "")
        words = words.replace("Component", "").replace("Widget", "")
        words = ''.join(c for c in words if c.isalnum() or c.isspace())
        parts = [w.capitalize() for w in words.split() if len(w) > 2]
        return ''.join(parts) if parts else "Component"
    
    def _make_display_name(self, component: str) -> str:
        """Convert PascalCase to display name with spaces"""
        import re
        return re.sub(r'([A-Z])', r' \1', component).strip()

# ============================================================================
# Main Generator Class
# ============================================================================

class JiraToCodeGenerator:
    """Main class orchestrating code generation"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.real_generator = RealCodeGenerator(self.config)
        self.mock_generator = MockCodeGenerator(self.config)
        self.prompt_builder = PromptBuilder()
    
    def generate_from_dump(self, dump_path: str, issue_keys: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate code for issues in a JIRA dump
        
        Args:
            dump_path: Path to JIRA dump JSON file
            issue_keys: Optional list of specific issue keys to generate (None = all)
            
        Returns:
            Dict mapping issue keys to generated code
        """
        parser = JiraParser(dump_path)
        issues = parser.get_issues()
        
        if issue_keys:
            issues = [i for i in issues if i.get("key") in issue_keys]
        
        results = {}
        for issue in issues:
            key = issue.get("key")
            print("\n" + "="*60)
            print("Generating code for: " + key + " - " + issue.get('summary'))
            print("="*60)
            
            code = self.generate_for_issue(issue, parser)
            results[key] = code
            
            # Save to file
            self._save_code(key, code, parser.detect_code_type(issue))
        
        return results
    
    def generate_for_issue(self, issue: Dict[str, Any], parser: JiraParser) -> str:
        """Generate code for a single issue"""
        code_type = parser.detect_code_type(issue)
        print("Detected code type: " + code_type)
        
        generated_code = ""
        
        # Try real generation first if mode allows
        if self.config.mode in [GenerationMode.REAL, GenerationMode.AUTO]:
            if self.real_generator.is_available():
                try:
                    print("Using REAL LLM generation...")
                    prompt = (self.prompt_builder.build_frontend_prompt(issue) 
                             if code_type == "frontend" 
                             else self.prompt_builder.build_backend_prompt(issue))
                    
                    for chunk in self.real_generator.generate(prompt):
                        print(chunk, end="", flush=True)
                        generated_code += chunk
                    print()
                    return generated_code
                except Exception as e:
                    print("Real generation failed: " + str(e))
                    if self.config.mode == GenerationMode.REAL:
                        raise
        
        # Fall back to mock generation
        print("Using MOCK template generation...")
        for chunk in self.mock_generator.generate(issue, code_type):
            print(chunk, end="", flush=True)
            generated_code += chunk
        print()
        
        return generated_code
    
    def _save_code(self, key: str, code: str, code_type: str):
        """Save generated code to file"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ext = "tsx" if code_type == "frontend" else "py"
        filename = key.lower().replace('-', '_') + "." + ext
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        print("\n✅ Saved to: " + str(filepath))

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Generate code from JIRA dump")
    arg_parser.add_argument("dump", help="Path to JIRA dump JSON file")
    arg_parser.add_argument("--issues", "-i", nargs="+", help="Specific issue keys to generate")
    arg_parser.add_argument("--mode", "-m", choices=["real", "mock", "auto"], default="auto",
                       help="Generation mode (default: auto)")
    arg_parser.add_argument("--output", "-o", default="./generated_code", help="Output directory")
    
    args = arg_parser.parse_args()
    
    config = Config(
        mode=GenerationMode(args.mode),
        output_dir=args.output
    )
    
    generator = JiraToCodeGenerator(config)
    results = generator.generate_from_dump(args.dump, args.issues)
    
    print("\n" + "="*60)
    print("✨ Generated code for " + str(len(results)) + " issues")
    print("Output directory: " + config.output_dir)
    print("="*60)

if __name__ == "__main__":
    main()