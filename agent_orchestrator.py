"""
Agent Orchestrator
Handles the complete workflow: JIRA -> Code Generation -> GitHub PR
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, Generator, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Suppress SSL warnings for corporate environments
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from github_client import GitHubClient, GitHubConfig, create_github_client
from jira_to_code import (
    JiraParser, JiraToCodeGenerator, Config, GenerationMode,
    MockCodeGenerator, PromptBuilder
)


class WorkflowStep(Enum):
    """Workflow steps for tracking progress"""
    INIT = "init"
    PARSE_JIRA = "parse_jira"
    GENERATE_CODE = "generate_code"
    CREATE_BRANCH = "create_branch"
    CREATE_ISSUE = "create_issue"
    COMMIT_FILES = "commit_files"
    CREATE_PR = "create_pr"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class WorkflowEvent:
    """Event emitted during workflow execution"""
    step: WorkflowStep
    status: str  # "started", "progress", "completed", "error"
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp
        }
    
    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class WorkflowConfig:
    """Configuration for the agent workflow"""
    github_token: Optional[str] = None
    github_repo: Optional[str] = None
    generation_mode: GenerationMode = GenerationMode.MOCK
    branch_prefix: str = "feature/agent"
    create_draft_pr: bool = False
    
    @classmethod
    def from_env(cls) -> "WorkflowConfig":
        return cls(
            github_token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT"),
            github_repo=os.environ.get("GITHUB_REPO"),
            generation_mode=GenerationMode.MOCK
        )


@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    success: bool
    jira_key: str
    branch_name: Optional[str] = None
    issue_number: Optional[int] = None
    issue_url: Optional[str] = None
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    files_created: list = field(default_factory=list)
    error: Optional[str] = None


class AgentOrchestrator:
    """
    Orchestrates the complete agent workflow:
    1. Parse JIRA ticket
    2. Generate code based on requirements
    3. Create feature branch
    4. Create GitHub issue
    5. Commit generated code
    6. Create pull request
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig.from_env()
        self.github_client: Optional[GitHubClient] = None
        self.code_generator = JiraToCodeGenerator(Config(mode=self.config.generation_mode))
    
    def _init_github(self) -> GitHubClient:
        """Initialize GitHub client"""
        if not self.config.github_token:
            raise ValueError("GitHub token not configured")
        if not self.config.github_repo:
            raise ValueError("GitHub repository not configured")
        
        gh_config = GitHubConfig(
            token=self.config.github_token,
            owner=self.config.github_repo.split("/")[0],
            repo=self.config.github_repo.split("/")[1]
        )
        return GitHubClient(gh_config)
    
    def _generate_branch_name(self, jira_key: str) -> str:
        """Generate a branch name from JIRA key"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{self.config.branch_prefix}/{jira_key.lower()}-{timestamp}"
    
    def _build_issue_body(self, issue: Dict[str, Any]) -> str:
        """Build GitHub issue body from JIRA data"""
        ac_list = "\n".join(f"- [ ] {c}" for c in issue.get("acceptance_criteria", []))
        labels = ", ".join(f"`{l}`" for l in issue.get("labels", []))
        tech = ", ".join(f"`{t}`" for t in issue.get("tech_stack", []))
        
        return f"""## JIRA Ticket: {issue.get('key')}

### Summary
{issue.get('summary')}

### Description
{issue.get('description', 'No description provided.')}

### Acceptance Criteria
{ac_list if ac_list else 'No acceptance criteria defined.'}

### Technical Details
- **Priority:** {issue.get('priority', 'Medium')}
- **Labels:** {labels if labels else 'None'}
- **Tech Stack:** {tech if tech else 'Not specified'}

---
*This issue was automatically created by Agent EngineerX*
"""
    
    def _build_pr_body(
        self,
        issue: Dict[str, Any],
        issue_number: int,
        files: Dict[str, str]
    ) -> str:
        """Build pull request body"""
        file_list = "\n".join(f"- `{f}`" for f in files.keys())
        
        return f"""## Summary
Automated code generation for JIRA ticket **{issue.get('key')}**

Closes #{issue_number}

## Changes
{issue.get('summary')}

## Files Created
{file_list}

## JIRA Details
- **Ticket:** {issue.get('key')}
- **Priority:** {issue.get('priority', 'Medium')}
- **Type:** {issue.get('type', 'Story')}

## Acceptance Criteria
{chr(10).join('- [ ] ' + c for c in issue.get('acceptance_criteria', []))}

---
*This PR was automatically created by Agent EngineerX*
"""
    
    async def execute_workflow(
        self,
        jira_issue: Dict[str, Any]
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Execute the complete workflow with streaming events
        
        Args:
            jira_issue: JIRA issue data dictionary
            
        Yields:
            WorkflowEvent objects for each step
        """
        result = WorkflowResult(success=False, jira_key=jira_issue.get("key", "UNKNOWN"))
        
        try:
            # Step 1: Initialize
            yield WorkflowEvent(
                step=WorkflowStep.INIT,
                status="started",
                message="Initializing agent workflow..."
            )
            
            self.github_client = self._init_github()
            repo_info = self.github_client.get_repo_info()
            
            yield WorkflowEvent(
                step=WorkflowStep.INIT,
                status="completed",
                message=f"Connected to repository: {repo_info['full_name']}",
                data={"repo": repo_info["full_name"], "default_branch": repo_info["default_branch"]}
            )
            
            # Step 2: Parse JIRA
            yield WorkflowEvent(
                step=WorkflowStep.PARSE_JIRA,
                status="started",
                message=f"Processing JIRA ticket: {jira_issue.get('key')}"
            )
            
            # Create a temporary parser for code type detection
            parser = JiraParser.__new__(JiraParser)
            parser.data = {"issues": [jira_issue]}
            code_type = parser.detect_code_type(jira_issue)
            
            yield WorkflowEvent(
                step=WorkflowStep.PARSE_JIRA,
                status="completed",
                message=f"Detected code type: {code_type}",
                data={
                    "key": jira_issue.get("key"),
                    "summary": jira_issue.get("summary"),
                    "code_type": code_type
                }
            )
            
            # Step 3: Generate Code
            yield WorkflowEvent(
                step=WorkflowStep.GENERATE_CODE,
                status="started",
                message="Generating code with Azure OpenAI..."
            )
            
            # Generate code using Azure OpenAI (REAL mode)
            from jira_to_code import RealCodeGenerator, MockCodeGenerator, Config as GenConfig, GenerationMode, PromptBuilder
            
            gen_config = GenConfig(mode=GenerationMode.REAL)
            
            # Try real LLM first, fallback to mock
            try:
                real_gen = RealCodeGenerator(gen_config)
                prompt_builder = PromptBuilder()
                
                if code_type == "frontend":
                    prompt = prompt_builder.build_frontend_prompt(jira_issue)
                else:
                    prompt = prompt_builder.build_backend_prompt(jira_issue)
                
                generated_code = ""
                chunk_count = 0
                
                for chunk in real_gen.generate(prompt):
                    generated_code += chunk
                    chunk_count += 1
                    if chunk_count % 20 == 0:
                        yield WorkflowEvent(
                            step=WorkflowStep.GENERATE_CODE,
                            status="progress",
                            message=f"Generating with Azure OpenAI... ({len(generated_code)} chars)",
                            data={"chars": len(generated_code)}
                        )
                    await asyncio.sleep(0.01)
                
                yield WorkflowEvent(
                    step=WorkflowStep.GENERATE_CODE,
                    status="progress",
                    message="✅ Code generated with Azure OpenAI",
                    data={"source": "azure_openai"}
                )
                
            except Exception as llm_error:
                yield WorkflowEvent(
                    step=WorkflowStep.GENERATE_CODE,
                    status="progress",
                    message=f"⚠️ LLM failed ({str(llm_error)[:50]}), using mock...",
                    data={"error": str(llm_error)}
                )
                
                # Fallback to mock
                mock_gen = MockCodeGenerator(gen_config)
                generated_code = ""
                for chunk in mock_gen.generate(jira_issue, code_type):
                    generated_code += chunk
                    await asyncio.sleep(0.01)
            
            # Determine file path
            file_ext = "tsx" if code_type == "frontend" else "py"
            file_name = jira_issue.get("key", "generated").lower().replace("-", "_") + f".{file_ext}"
            
            if code_type == "frontend":
                file_path = f"src/components/{file_name}"
            else:
                file_path = f"src/api/{file_name}"
            
            files_to_create = {file_path: generated_code}
            result.files_created = list(files_to_create.keys())
            
            yield WorkflowEvent(
                step=WorkflowStep.GENERATE_CODE,
                status="completed",
                message=f"Generated {len(generated_code.splitlines())} lines of code",
                data={
                    "file": file_path,
                    "lines": len(generated_code.splitlines()),
                    "code_preview": generated_code[:500] + "..." if len(generated_code) > 500 else generated_code
                }
            )
            
            # Step 4: Create Branch
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_BRANCH,
                status="started",
                message="Creating feature branch..."
            )
            
            branch_name = self._generate_branch_name(jira_issue.get("key"))
            default_branch = self.github_client.get_default_branch()
            
            self.github_client.create_branch(branch_name, default_branch)
            result.branch_name = branch_name
            
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_BRANCH,
                status="completed",
                message=f"Created branch: {branch_name}",
                data={"branch": branch_name, "from": default_branch}
            )
            
            # Step 5: Create Issue
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_ISSUE,
                status="started",
                message="Creating GitHub issue..."
            )
            
            issue_title = f"[{jira_issue.get('key')}] {jira_issue.get('summary')}"
            issue_body = self._build_issue_body(jira_issue)
            
            gh_issue = self.github_client.create_issue(
                title=issue_title,
                body=issue_body,
                labels=jira_issue.get("labels", [])
            )
            
            result.issue_number = gh_issue["number"]
            result.issue_url = gh_issue["html_url"]
            
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_ISSUE,
                status="completed",
                message=f"Created issue #{gh_issue['number']}",
                data={"issue_number": gh_issue["number"], "issue_url": gh_issue["html_url"]}
            )
            
            # Step 6: Commit Files
            yield WorkflowEvent(
                step=WorkflowStep.COMMIT_FILES,
                status="started",
                message=f"Committing {len(files_to_create)} file(s)..."
            )
            
            commit_message = f"feat({jira_issue.get('key')}): {jira_issue.get('summary')}\n\nCloses #{gh_issue['number']}"
            
            commit = self.github_client.create_files_batch(
                files=files_to_create,
                message=commit_message,
                branch=branch_name
            )
            
            yield WorkflowEvent(
                step=WorkflowStep.COMMIT_FILES,
                status="completed",
                message=f"Committed files to {branch_name}",
                data={"commit_sha": commit["sha"][:7], "files": list(files_to_create.keys())}
            )
            
            # Step 7: Create PR
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_PR,
                status="started",
                message="Creating pull request..."
            )
            
            pr_title = f"[{jira_issue.get('key')}] {jira_issue.get('summary')}"
            pr_body = self._build_pr_body(jira_issue, gh_issue["number"], files_to_create)
            
            pr = self.github_client.create_pull_request(
                title=pr_title,
                body=pr_body,
                head_branch=branch_name,
                base_branch=default_branch,
                draft=self.config.create_draft_pr
            )
            
            result.pr_number = pr["number"]
            result.pr_url = pr["html_url"]
            
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_PR,
                status="completed",
                message=f"Created PR #{pr['number']}",
                data={"pr_number": pr["number"], "pr_url": pr["html_url"]}
            )
            
            # Complete!
            result.success = True
            
            yield WorkflowEvent(
                step=WorkflowStep.COMPLETE,
                status="completed",
                message="🎉 Workflow completed successfully!",
                data={
                    "jira_key": result.jira_key,
                    "branch": result.branch_name,
                    "issue_number": result.issue_number,
                    "issue_url": result.issue_url,
                    "pr_number": result.pr_number,
                    "pr_url": result.pr_url,
                    "files": result.files_created
                }
            )
            
        except Exception as e:
            result.error = str(e)
            yield WorkflowEvent(
                step=WorkflowStep.ERROR,
                status="error",
                message=f"Workflow failed: {str(e)}",
                data={"error": str(e), "result": result.__dict__}
            )
    
    def execute_workflow_sync(self, jira_issue: Dict[str, Any]) -> Generator[WorkflowEvent, None, None]:
        """Synchronous version of execute_workflow for non-async contexts"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def collect_events():
            events = []
            async for event in self.execute_workflow(jira_issue):
                events.append(event)
            return events
        
        try:
            events = loop.run_until_complete(collect_events())
            for event in events:
                yield event
        finally:
            loop.close()


# =========================================================================
# Demo / Test
# =========================================================================

def demo():
    """Run a demo of the agent workflow"""
    print("\n" + "="*60)
    print("🤖 Agent Orchestrator Demo")
    print("="*60)
    
    # Check configuration
    config = WorkflowConfig.from_env()
    
    if not config.github_token:
        print("\n❌ GITHUB_TOKEN not set")
        print("   export GITHUB_TOKEN=your_token")
        return
    
    if not config.github_repo:
        print("\n❌ GITHUB_REPO not set")
        print("   export GITHUB_REPO=owner/repo")
        return
    
    print(f"\n✅ Configuration OK")
    print(f"   Repository: {config.github_repo}")
    
    # Sample JIRA issue
    jira_issue = {
        "key": "DEMO-999",
        "type": "Story",
        "summary": "Create User Profile API",
        "description": "Implement REST API endpoints for user profile management",
        "acceptance_criteria": [
            "GET /api/users/{id} returns user profile",
            "PUT /api/users/{id} updates user profile",
            "Validate all input fields"
        ],
        "labels": ["backend", "api"],
        "tech_stack": ["Python", "FastAPI"],
        "priority": "High"
    }
    
    print(f"\n📋 Processing: {jira_issue['key']} - {jira_issue['summary']}")
    print("-"*60)
    
    # Execute workflow
    orchestrator = AgentOrchestrator(config)
    
    for event in orchestrator.execute_workflow_sync(jira_issue):
        emoji = {
            "started": "🔄",
            "progress": "⏳",
            "completed": "✅",
            "error": "❌"
        }.get(event.status, "•")
        
        print(f"{emoji} [{event.step.value}] {event.message}")
        
        if event.step == WorkflowStep.COMPLETE:
            print("\n" + "="*60)
            print("📊 Results:")
            print(f"   Branch: {event.data.get('branch')}")
            print(f"   Issue: {event.data.get('issue_url')}")
            print(f"   PR: {event.data.get('pr_url')}")
            print("="*60)


if __name__ == "__main__":
    demo()