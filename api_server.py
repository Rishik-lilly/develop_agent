"""
FastAPI Server for JIRA to Code Generator + Agent Workflow
Provides REST API + SSE streaming for real-time code generation and GitHub automation
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import asyncio
import os

# Suppress SSL warnings for corporate environments
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from jira_to_code import (
    JiraToCodeGenerator, JiraParser, Config, GenerationMode,
    MockCodeGenerator, RealCodeGenerator, PromptBuilder
)
from github_client import GitHubClient, GitHubConfig, create_github_client
from agent_orchestrator import AgentOrchestrator, WorkflowConfig, WorkflowEvent

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="JIRA to Code Agent API",
    description="Generate code from JIRA tickets and create GitHub PRs automatically",
    version="2.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Models
# ============================================================================

class JiraIssue(BaseModel):
    key: str
    type: str = "Story"
    summary: str
    description: str = ""
    acceptance_criteria: List[str] = []
    labels: List[str] = []
    tech_stack: List[str] = []
    priority: str = "Medium"
    status: str = "To Do"
    assignee: str = ""
    estimated_hours: int = 0
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "key": "DEMO-101",
                "type": "Story",
                "summary": "Create User Authentication API",
                "description": "Implement secure auth system",
                "acceptance_criteria": ["POST /api/auth/login accepts email and password"],
                "labels": ["backend", "api"],
                "tech_stack": ["Python", "FastAPI"],
                "priority": "High"
            }
        }
    }

class GenerateRequest(BaseModel):
    issue: JiraIssue
    mode: str = "mock"
    stream: bool = True

class AgentWorkflowRequest(BaseModel):
    issue: JiraIssue
    github_repo: str
    github_token: Optional[str] = None  # If not provided, uses env var
    create_draft_pr: bool = False

class GitHubConfigRequest(BaseModel):
    github_repo: str
    github_token: Optional[str] = None

# ============================================================================
# Config Helper
# ============================================================================

def get_config(mode: str = "mock") -> Config:
    return Config(
        mode=GenerationMode(mode),
        api_key=os.environ.get("CORTEX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )

def get_workflow_config(
    github_repo: str,
    github_token: Optional[str] = None,
    create_draft_pr: bool = False
) -> WorkflowConfig:
    return WorkflowConfig(
        github_token=github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT"),
        github_repo=github_repo,
        generation_mode=GenerationMode.MOCK,
        create_draft_pr=create_draft_pr
    )

# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "service": "JIRA to Code Agent",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "GET /issues": "List issues from JIRA dump",
            "GET /issues/{key}": "Get specific issue details",
            "POST /generate": "Generate code for an issue",
            "GET /generate/stream/{key}": "Stream code generation (SSE)",
            "POST /agent/workflow": "Execute full agent workflow",
            "GET /agent/workflow/stream": "Stream agent workflow (SSE)",
            "GET /github/test": "Test GitHub connection",
        }
    }

@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "healthy"}

# ============================================================================
# JIRA Dump Endpoints
# ============================================================================

@app.get("/issues")
async def list_issues(dump_path: str = Query(default="sample_jira_dump.json")):
    """List all issues from JIRA dump"""
    try:
        parser = JiraParser(dump_path)
        issues = parser.get_issues()
        
        return {
            "total": len(issues),
            "issues": [
                {
                    "key": i.get("key"),
                    "summary": i.get("summary"),
                    "type": i.get("type"),
                    "priority": i.get("priority"),
                    "status": i.get("status"),
                    "labels": i.get("labels", []),
                    "code_type": parser.detect_code_type(i)
                }
                for i in issues
            ]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dump file not found: " + dump_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/issues/{key}")
async def get_issue(key: str, dump_path: str = Query(default="sample_jira_dump.json")):
    """Get detailed issue information"""
    try:
        parser = JiraParser(dump_path)
        issue = parser.get_issue_by_key(key)
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found: " + key)
        
        return {
            **issue,
            "detected_code_type": parser.detect_code_type(issue)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Code Generation Endpoints
# ============================================================================

@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Generate code for a JIRA issue (non-streaming)"""
    config = get_config(request.mode)
    mock_gen = MockCodeGenerator(config)
    
    issue_dict = request.issue.model_dump()
    parser = JiraParser.__new__(JiraParser)
    parser.data = {"issues": [issue_dict]}
    code_type = parser.detect_code_type(issue_dict)
    
    code_chunks = []
    for chunk in mock_gen.generate(issue_dict, code_type):
        code_chunks.append(chunk)
    
    full_code = "".join(code_chunks)
    
    return {
        "issue_key": request.issue.key,
        "code_type": code_type,
        "mode": request.mode,
        "code": full_code,
        "lines": len(full_code.split("\n"))
    }

@app.get("/generate/stream/{key}")
async def stream_generate_code(
    key: str,
    dump_path: str = Query(default="sample_jira_dump.json"),
    mode: str = Query(default="mock")
):
    """Stream code generation using Server-Sent Events (SSE)"""
    
    async def event_generator():
        try:
            yield "data: " + json.dumps({'type': 'start', 'issue_key': key}) + "\n\n"
            
            parser = JiraParser(dump_path)
            issue = parser.get_issue_by_key(key)
            
            if not issue:
                yield "data: " + json.dumps({'type': 'error', 'message': 'Issue not found: ' + key}) + "\n\n"
                return
            
            code_type = parser.detect_code_type(issue)
            yield "data: " + json.dumps({'type': 'info', 'code_type': code_type, 'summary': issue.get('summary')}) + "\n\n"
            
            config = get_config(mode)
            mock_gen = MockCodeGenerator(config)
            
            full_code = ""
            for chunk in mock_gen.generate(issue, code_type):
                full_code += chunk
                yield "data: " + json.dumps({'type': 'chunk', 'content': chunk}) + "\n\n"
                await asyncio.sleep(0.01)
            
            yield "data: " + json.dumps({'type': 'complete', 'total_lines': len(full_code.split('\n'))}) + "\n\n"
            
        except Exception as e:
            yield "data: " + json.dumps({'type': 'error', 'message': str(e)}) + "\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================================
# GitHub Endpoints
# ============================================================================

@app.post("/github/test")
async def test_github_connection(config: GitHubConfigRequest):
    """Test GitHub connection and permissions"""
    try:
        token = config.github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        
        if not token:
            raise HTTPException(status_code=400, detail="GitHub token required")
        
        parts = config.github_repo.split("/")
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid repo format. Use: owner/repo")
        
        gh_config = GitHubConfig(token=token, owner=parts[0], repo=parts[1])
        client = GitHubClient(gh_config)
        
        repo_info = client.get_repo_info()
        
        return {
            "status": "connected",
            "repo": repo_info["full_name"],
            "default_branch": repo_info["default_branch"],
            "permissions": repo_info.get("permissions", {}),
            "private": repo_info["private"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="GitHub connection failed: " + str(e))

# ============================================================================
# Agent Workflow Endpoints
# ============================================================================

@app.post("/agent/workflow")
async def execute_agent_workflow(request: AgentWorkflowRequest):
    """
    Execute the complete agent workflow (non-streaming).
    Creates branch, issue, commits code, and creates PR.
    """
    try:
        config = get_workflow_config(
            github_repo=request.github_repo,
            github_token=request.github_token,
            create_draft_pr=request.create_draft_pr
        )
        
        orchestrator = AgentOrchestrator(config)
        issue_dict = request.issue.model_dump()
        
        result = None
        events = []
        
        async for event in orchestrator.execute_workflow(issue_dict):
            events.append(event.to_dict())
            if event.step.value == "complete" or event.step.value == "error":
                result = event.data
        
        return {
            "success": result.get("pr_url") is not None if result else False,
            "events": events,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/workflow/stream")
async def stream_agent_workflow(
    key: str = Query(..., description="JIRA issue key"),
    github_repo: str = Query(..., description="GitHub repo (owner/repo)"),
    github_token: Optional[str] = Query(default=None, description="GitHub token (uses env if not provided)"),
    dump_path: str = Query(default="sample_jira_dump.json")
):
    """
    Stream the agent workflow execution using SSE.
    Real-time updates as the agent creates branch, issue, code, and PR.
    """
    
    async def event_generator():
        try:
            # Load issue from dump
            yield "data: " + json.dumps({
                'type': 'workflow_start',
                'message': 'Starting agent workflow...',
                'issue_key': key
            }) + "\n\n"
            
            parser = JiraParser(dump_path)
            issue = parser.get_issue_by_key(key)
            
            if not issue:
                yield "data: " + json.dumps({
                    'type': 'error',
                    'message': 'Issue not found: ' + key
                }) + "\n\n"
                return
            
            # Configure and run workflow
            config = get_workflow_config(
                github_repo=github_repo,
                github_token=github_token
            )
            
            orchestrator = AgentOrchestrator(config)
            
            # Stream workflow events
            async for event in orchestrator.execute_workflow(issue):
                yield event.to_sse()
                await asyncio.sleep(0.05)  # Small delay for visual effect
            
        except Exception as e:
            yield "data: " + json.dumps({
                'type': 'error',
                'message': 'Workflow error: ' + str(e)
            }) + "\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/agent/workflow/custom")
async def stream_custom_workflow(request: AgentWorkflowRequest):
    """
    Execute agent workflow with a custom JIRA issue (streaming via SSE).
    Use this when you want to provide issue details directly instead of from dump.
    """
    
    async def event_generator():
        try:
            issue_dict = request.issue.model_dump()
            
            yield "data: " + json.dumps({
                'type': 'workflow_start',
                'message': 'Starting agent workflow with custom issue...',
                'issue': {
                    'key': issue_dict['key'],
                    'summary': issue_dict['summary']
                }
            }) + "\n\n"
            
            config = get_workflow_config(
                github_repo=request.github_repo,
                github_token=request.github_token,
                create_draft_pr=request.create_draft_pr
            )
            
            orchestrator = AgentOrchestrator(config)
            
            async for event in orchestrator.execute_workflow(issue_dict):
                yield event.to_sse()
                await asyncio.sleep(0.05)
            
        except Exception as e:
            yield "data: " + json.dumps({
                'type': 'error',
                'message': 'Workflow error: ' + str(e)
            }) + "\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🤖 JIRA to Code Agent - API Server")
    print("="*70)
    print("\n📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n💡 Quick Tests:")
    print("   curl http://localhost:8000/issues")
    print("   curl http://localhost:8000/generate/stream/DEMO-101")
    print("\n🔧 For Agent Workflow, set environment variables:")
    print("   export GITHUB_TOKEN=your_token")
    print("   export GITHUB_REPO=owner/repo")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)