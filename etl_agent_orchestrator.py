"""
GitHub ETL Agent - Complete Implementation
==========================================

ETL Agent that connects to GitHub for data pipeline automation.
Following the Agent EngineerX orchestrator pattern.

Architecture:
    GitHub Source → ETL Agent → Transform → GitHub PR (with generated code)
"""

import os
import json
import logging
import base64
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Suppress SSL warnings for corporate environments
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ETLTaskType(Enum):
    """Types of ETL tasks the agent can handle"""
    SCHEMA_EXTRACTION = "schema_extraction"
    DATA_MIGRATION = "data_migration"
    SQL_GENERATION = "sql_generation"
    PIPELINE_CREATION = "pipeline_creation"
    CONFIG_SYNC = "config_sync"
    UNKNOWN = "unknown"


class TaskComplexity(Enum):
    """Task complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WorkflowStep(Enum):
    """ETL Workflow steps for tracking progress"""
    INIT = "init"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    GENERATE_CODE = "generate_code"
    CREATE_BRANCH = "create_branch"
    COMMIT_FILES = "commit_files"
    CREATE_PR = "create_pr"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ETLContext:
    """Context information passed through ETL pipeline"""
    task_id: str
    source_repo: str
    target_repo: Optional[str]
    task_type: ETLTaskType
    task_complexity: TaskComplexity
    source_files: List[str] = None
    target_schema: Optional[str] = None
    config: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.source_files is None:
            self.source_files = []
        if self.config is None:
            self.config = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ETLResponse:
    """Standardized response from ETL agents"""
    success: bool
    task_type: ETLTaskType
    result: Dict[str, Any]
    reasoning: str
    confidence_score: float
    generated_files: Dict[str, str] = None
    next_actions: List[str] = None
    warnings: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.generated_files is None:
            self.generated_files = {}
        if self.next_actions is None:
            self.next_actions = []
        if self.warnings is None:
            self.warnings = []
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


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


# =============================================================================
# GITHUB CLIENT
# =============================================================================

@dataclass
class GitHubConfig:
    """GitHub configuration"""
    token: str
    owner: str
    repo: str
    base_url: str = "https://api.github.com"
    proxy: Optional[str] = None
    verify_ssl: bool = False
    
    @classmethod
    def from_env(cls, repo_full_name: Optional[str] = None) -> "GitHubConfig":
        """Create config from environment variables"""
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        if not token:
            raise ValueError("GITHUB_TOKEN or GITHUB_PAT environment variable required")
        
        repo_name = repo_full_name or os.environ.get("GITHUB_REPO", "")
        if "/" in repo_name:
            owner, repo = repo_name.split("/", 1)
        else:
            owner = os.environ.get("GITHUB_OWNER", "")
            repo = repo_name
        
        proxy = (
            os.environ.get("HTTPS_PROXY") or 
            os.environ.get("https_proxy") or
            os.environ.get("HTTP_PROXY") or
            os.environ.get("http_proxy")
        )
        
        return cls(token=token, owner=owner, repo=repo, proxy=proxy)


class GitHubClient:
    """GitHub API client for ETL agent operations"""
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.base_url = config.base_url
        self.repo_path = f"/repos/{config.owner}/{config.repo}"
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.verify = config.verify_ssl
        
        if config.proxy:
            self.session.proxies = {"http": config.proxy, "https": config.proxy}
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to GitHub API"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}
    
    def get_repo_info(self) -> Dict[str, Any]:
        """Get repository information"""
        return self._request("GET", self.repo_path)
    
    def get_default_branch(self) -> str:
        """Get the default branch name"""
        repo = self.get_repo_info()
        return repo["default_branch"]
    
    def get_branch_sha(self, branch_name: str) -> str:
        """Get the SHA of a branch's HEAD"""
        branch = self._request("GET", f"{self.repo_path}/branches/{branch_name}")
        return branch["commit"]["sha"]
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> Dict[str, Any]:
        """Create a new branch"""
        if from_branch is None:
            from_branch = self.get_default_branch()
        sha = self.get_branch_sha(from_branch)
        return self._request(
            "POST", f"{self.repo_path}/git/refs",
            json={"ref": f"refs/heads/{branch_name}", "sha": sha}
        )
    
    def delete_branch(self, branch_name: str) -> None:
        """Delete a branch"""
        self._request("DELETE", f"{self.repo_path}/git/refs/heads/{branch_name}")
    
    def get_file_content(self, file_path: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Get file content from repository"""
        params = {"ref": branch} if branch else {}
        return self._request("GET", f"{self.repo_path}/contents/{file_path}", params=params)
    
    def get_file_contents_decoded(self, file_path: str, branch: Optional[str] = None) -> str:
        """Get decoded file content"""
        content_data = self.get_file_content(file_path, branch)
        return base64.b64decode(content_data["content"]).decode("utf-8")
    
    def list_directory(self, path: str = "", branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """List contents of a directory"""
        params = {"ref": branch} if branch else {}
        return self._request("GET", f"{self.repo_path}/contents/{path}", params=params)
    
    def search_files(self, query: str, extension: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for files in repository"""
        q = f"{query} repo:{self.config.owner}/{self.config.repo}"
        if extension:
            q += f" extension:{extension}"
        result = self._request("GET", "/search/code", params={"q": q})
        return result.get("items", [])
    
    def create_files_batch(
        self, files: Dict[str, str], message: str, branch: str
    ) -> Dict[str, Any]:
        """Create multiple files in a single commit"""
        branch_sha = self.get_branch_sha(branch)
        
        # Get base tree
        commit = self._request("GET", f"{self.repo_path}/git/commits/{branch_sha}")
        base_tree = commit["tree"]["sha"]
        
        # Create blobs for each file
        tree_items = []
        for path, content in files.items():
            blob = self._request(
                "POST", f"{self.repo_path}/git/blobs",
                json={"content": content, "encoding": "utf-8"}
            )
            tree_items.append({
                "path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]
            })
        
        # Create new tree
        new_tree = self._request(
            "POST", f"{self.repo_path}/git/trees",
            json={"base_tree": base_tree, "tree": tree_items}
        )
        
        # Create commit
        new_commit = self._request(
            "POST", f"{self.repo_path}/git/commits",
            json={"message": message, "tree": new_tree["sha"], "parents": [branch_sha]}
        )
        
        # Update branch reference
        self._request(
            "PATCH", f"{self.repo_path}/git/refs/heads/{branch}",
            json={"sha": new_commit["sha"]}
        )
        
        return new_commit
    
    def create_pull_request(
        self, title: str, body: str, head_branch: str,
        base_branch: Optional[str] = None, draft: bool = False
    ) -> Dict[str, Any]:
        """Create a pull request"""
        if base_branch is None:
            base_branch = self.get_default_branch()
        return self._request(
            "POST", f"{self.repo_path}/pulls",
            json={
                "title": title, "body": body, "head": head_branch,
                "base": base_branch, "draft": draft
            }
        )


def create_github_client(repo: Optional[str] = None) -> GitHubClient:
    """Create a GitHub client from environment"""
    config = GitHubConfig.from_env(repo)
    return GitHubClient(config)


# =============================================================================
# BASE ETL AGENT
# =============================================================================

class BaseETLAgent(ABC):
    """Base class for all ETL agents"""
    
    def __init__(self, name: str, task_type: ETLTaskType):
        self.name = name
        self.task_type = task_type
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def can_handle(self, context: ETLContext) -> bool:
        """Determine if this agent can handle the given context"""
        pass
    
    @abstractmethod
    def execute(self, context: ETLContext) -> ETLResponse:
        """Execute the agent's main functionality"""
        pass
    
    def validate_context(self, context: ETLContext) -> bool:
        """Validate the context before processing"""
        if not context.task_id:
            self.logger.error("Missing task_id in context")
            return False
        if not context.source_repo:
            self.logger.error("Missing source_repo in context")
            return False
        return True


class LLMIntegratedETLAgent(BaseETLAgent):
    """Base class for ETL agents with LLM integration"""
    
    def __init__(self, name: str, task_type: ETLTaskType, llm_service=None):
        super().__init__(name, task_type)
        self.llm_service = llm_service
    
    def invoke_llm(self, prompt: str, expects_json: bool = True) -> Any:
        """Invoke LLM with the given prompt"""
        if not self.llm_service:
            self.logger.warning("LLM service not configured, using mock response")
            return self._mock_llm_response(prompt, expects_json)
        try:
            return self.llm_service.invoke_model(prompt, expects_json)
        except Exception as e:
            self.logger.error(f"Error invoking LLM: {str(e)}")
            raise
    
    def _mock_llm_response(self, prompt: str, expects_json: bool) -> Any:
        """Mock LLM response for testing"""
        if expects_json:
            return {"status": "mock", "generated": True}
        return "Mock LLM response"


# =============================================================================
# SPECIALIZED ETL AGENTS
# =============================================================================

class SchemaExtractionAgent(LLMIntegratedETLAgent):
    """Agent for extracting and analyzing database schemas from GitHub repos"""
    
    def __init__(self, llm_service=None):
        super().__init__("SchemaExtractionAgent", ETLTaskType.SCHEMA_EXTRACTION, llm_service)
    
    def can_handle(self, context: ETLContext) -> bool:
        return context.task_type == ETLTaskType.SCHEMA_EXTRACTION
    
    def execute(self, context: ETLContext) -> ETLResponse:
        try:
            if not self.validate_context(context):
                return ETLResponse(
                    success=False, task_type=self.task_type, result={},
                    reasoning="Invalid context", confidence_score=0.0
                )
            
            self.logger.info(f"Extracting schemas from {context.source_repo}")
            
            # Extract schema files
            schemas = self._extract_schema_files(context)
            
            # Analyze schemas
            analysis = self._analyze_schemas(schemas)
            
            return ETLResponse(
                success=True, task_type=self.task_type,
                result={"schemas": schemas, "analysis": analysis},
                reasoning="Schema extraction completed successfully",
                confidence_score=0.95,
                next_actions=["Generate migration scripts", "Create data models"]
            )
        except Exception as e:
            return ETLResponse(
                success=False, task_type=self.task_type,
                result={"error": str(e)},
                reasoning=f"Schema extraction failed: {str(e)}",
                confidence_score=0.0
            )
    
    def _extract_schema_files(self, context: ETLContext) -> List[Dict[str, Any]]:
        """Extract schema definition files from repository"""
        schemas = []
        schema_extensions = [".sql", ".ddl", ".json", ".yaml", ".yml"]
        
        for file_path in context.source_files:
            if any(file_path.endswith(ext) for ext in schema_extensions):
                schemas.append({
                    "path": file_path,
                    "type": self._detect_schema_type(file_path),
                    "extracted": True
                })
        return schemas
    
    def _detect_schema_type(self, file_path: str) -> str:
        """Detect schema type from file extension"""
        if file_path.endswith((".sql", ".ddl")):
            return "sql"
        elif file_path.endswith(".json"):
            return "json_schema"
        elif file_path.endswith((".yaml", ".yml")):
            return "yaml_schema"
        return "unknown"
    
    def _analyze_schemas(self, schemas: List[Dict]) -> Dict[str, Any]:
        """Analyze extracted schemas"""
        return {
            "total_schemas": len(schemas),
            "by_type": {s["type"]: sum(1 for x in schemas if x["type"] == s["type"]) for s in schemas},
            "recommendations": ["Review schema compatibility", "Validate data types"]
        }


class SQLGenerationAgent(LLMIntegratedETLAgent):
    """Agent for generating SQL/ETL code from specifications"""
    
    def __init__(self, llm_service=None):
        super().__init__("SQLGenerationAgent", ETLTaskType.SQL_GENERATION, llm_service)
    
    def can_handle(self, context: ETLContext) -> bool:
        return context.task_type == ETLTaskType.SQL_GENERATION
    
    def execute(self, context: ETLContext) -> ETLResponse:
        try:
            if not self.validate_context(context):
                return ETLResponse(
                    success=False, task_type=self.task_type, result={},
                    reasoning="Invalid context", confidence_score=0.0
                )
            
            self.logger.info(f"Generating SQL for task {context.task_id}")
            
            # Generate SQL code
            generated_files = self._generate_sql(context)
            
            return ETLResponse(
                success=True, task_type=self.task_type,
                result={"files_generated": len(generated_files)},
                generated_files=generated_files,
                reasoning="SQL generation completed successfully",
                confidence_score=0.9,
                next_actions=["Review generated SQL", "Run validation tests"]
            )
        except Exception as e:
            return ETLResponse(
                success=False, task_type=self.task_type,
                result={"error": str(e)},
                reasoning=f"SQL generation failed: {str(e)}",
                confidence_score=0.0
            )
    
    def _generate_sql(self, context: ETLContext) -> Dict[str, str]:
        """Generate SQL files based on context"""
        target_schema = context.target_schema or "dbo"
        task_id = context.task_id.lower().replace("-", "_")
        
        # Generate stored procedure
        sp_content = self._generate_stored_procedure(context, target_schema, task_id)
        
        # Generate merge statement
        merge_content = self._generate_merge_statement(context, target_schema, task_id)
        
        return {
            f"etl/stored_procedures/sp_{task_id}.sql": sp_content,
            f"etl/dml/merge_{task_id}.sql": merge_content
        }
    
    def _generate_stored_procedure(self, ctx: ETLContext, schema: str, task_id: str) -> str:
        """Generate a stored procedure template"""
        return f'''-- ============================================================
-- ETL Stored Procedure: {task_id}
-- Generated by: GitHub ETL Agent
-- Task ID: {ctx.task_id}
-- Generated: {datetime.now(timezone.utc).isoformat()}
-- ============================================================

CREATE OR ALTER PROCEDURE [{schema}].[usp_ETL_{task_id}]
    @BatchDate DATE = NULL,
    @DebugMode BIT = 0
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    
    -- Initialize batch date
    IF @BatchDate IS NULL
        SET @BatchDate = CAST(GETDATE() AS DATE);
    
    DECLARE @RowsAffected INT = 0;
    DECLARE @ErrorMessage NVARCHAR(4000);
    DECLARE @StartTime DATETIME2 = SYSDATETIME();
    
    BEGIN TRY
        BEGIN TRANSACTION;
        
        -- ========================================
        -- EXTRACT: Source data extraction
        -- ========================================
        IF @DebugMode = 1
            PRINT 'Starting extraction phase...';
        
        -- TODO: Add extraction logic
        
        -- ========================================
        -- TRANSFORM: Data transformation
        -- ========================================
        IF @DebugMode = 1
            PRINT 'Starting transformation phase...';
        
        -- TODO: Add transformation logic
        
        -- ========================================
        -- LOAD: Target table population
        -- ========================================
        IF @DebugMode = 1
            PRINT 'Starting load phase...';
        
        -- TODO: Add load logic using MERGE
        
        COMMIT TRANSACTION;
        
        -- Log success
        IF @DebugMode = 1
        BEGIN
            PRINT 'ETL completed successfully.';
            PRINT 'Rows affected: ' + CAST(@RowsAffected AS VARCHAR(20));
            PRINT 'Duration: ' + CAST(DATEDIFF(SECOND, @StartTime, SYSDATETIME()) AS VARCHAR(20)) + ' seconds';
        END
        
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;
        
        SET @ErrorMessage = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END
GO
'''
    
    def _generate_merge_statement(self, ctx: ETLContext, schema: str, task_id: str) -> str:
        """Generate a MERGE statement template"""
        return f'''-- ============================================================
-- ETL MERGE Statement: {task_id}
-- Generated by: GitHub ETL Agent
-- Task ID: {ctx.task_id}
-- ============================================================

MERGE [{schema}].[Target_{task_id}] AS target
USING (
    SELECT 
        -- Source columns
        src.*
    FROM [{schema}].[Source_{task_id}] src
    WHERE src.IsActive = 1
) AS source
ON target.BusinessKey = source.BusinessKey

WHEN MATCHED AND (
    target.HashValue <> source.HashValue
    OR target.ModifiedDate < source.ModifiedDate
)
THEN UPDATE SET
    target.Column1 = source.Column1,
    target.Column2 = source.Column2,
    target.ModifiedDate = GETDATE(),
    target.ModifiedBy = SYSTEM_USER

WHEN NOT MATCHED BY TARGET
THEN INSERT (
    BusinessKey,
    Column1,
    Column2,
    CreatedDate,
    CreatedBy
)
VALUES (
    source.BusinessKey,
    source.Column1,
    source.Column2,
    GETDATE(),
    SYSTEM_USER
)

WHEN NOT MATCHED BY SOURCE AND target.IsActive = 1
THEN UPDATE SET
    target.IsActive = 0,
    target.DeactivatedDate = GETDATE();

-- Output results
OUTPUT $action AS MergeAction,
       inserted.BusinessKey AS InsertedKey,
       deleted.BusinessKey AS DeletedKey;
'''


class PipelineCreationAgent(LLMIntegratedETLAgent):
    """Agent for creating ETL pipeline configurations"""
    
    def __init__(self, llm_service=None):
        super().__init__("PipelineCreationAgent", ETLTaskType.PIPELINE_CREATION, llm_service)
    
    def can_handle(self, context: ETLContext) -> bool:
        return context.task_type == ETLTaskType.PIPELINE_CREATION
    
    def execute(self, context: ETLContext) -> ETLResponse:
        try:
            if not self.validate_context(context):
                return ETLResponse(
                    success=False, task_type=self.task_type, result={},
                    reasoning="Invalid context", confidence_score=0.0
                )
            
            self.logger.info(f"Creating pipeline for task {context.task_id}")
            
            # Generate pipeline files
            generated_files = self._generate_pipeline(context)
            
            return ETLResponse(
                success=True, task_type=self.task_type,
                result={"pipeline_created": True},
                generated_files=generated_files,
                reasoning="Pipeline creation completed",
                confidence_score=0.85,
                next_actions=["Configure connections", "Test pipeline"]
            )
        except Exception as e:
            return ETLResponse(
                success=False, task_type=self.task_type,
                result={"error": str(e)},
                reasoning=f"Pipeline creation failed: {str(e)}",
                confidence_score=0.0
            )
    
    def _generate_pipeline(self, context: ETLContext) -> Dict[str, str]:
        """Generate pipeline configuration files"""
        task_id = context.task_id.lower().replace("-", "_")
        
        # Generate Airflow DAG
        airflow_dag = self._generate_airflow_dag(context, task_id)
        
        # Generate pipeline config
        config = self._generate_pipeline_config(context, task_id)
        
        return {
            f"pipelines/dags/{task_id}_dag.py": airflow_dag,
            f"pipelines/config/{task_id}_config.yaml": config
        }
    
    def _generate_airflow_dag(self, ctx: ETLContext, task_id: str) -> str:
        """Generate Airflow DAG"""
        return f'''"""
ETL Pipeline DAG: {task_id}
Generated by: GitHub ETL Agent
Task ID: {ctx.task_id}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.sql import SQLExecuteQueryOperator
from airflow.utils.dates import days_ago

default_args = {{
    'owner': 'etl_agent',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}}

dag = DAG(
    dag_id='{task_id}_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline generated by GitHub ETL Agent',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'automated', '{ctx.task_id}'],
)

def extract_data(**context):
    """Extract data from source"""
    # TODO: Implement extraction logic
    pass

def transform_data(**context):
    """Transform extracted data"""
    # TODO: Implement transformation logic
    pass

def load_data(**context):
    """Load data to target"""
    # TODO: Implement load logic
    pass

# Task definitions
extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load_data,
    dag=dag,
)

# Task dependencies
extract_task >> transform_task >> load_task
'''
    
    def _generate_pipeline_config(self, ctx: ETLContext, task_id: str) -> str:
        """Generate pipeline configuration"""
        return f'''# ETL Pipeline Configuration
# Task ID: {ctx.task_id}
# Generated by: GitHub ETL Agent

pipeline:
  name: {task_id}
  version: "1.0.0"
  description: "ETL pipeline for {ctx.task_id}"

source:
  type: database
  connection: ${{SOURCE_CONNECTION_STRING}}
  schema: {ctx.target_schema or 'source'}

target:
  type: database
  connection: ${{TARGET_CONNECTION_STRING}}
  schema: {ctx.target_schema or 'target'}

extraction:
  mode: incremental
  watermark_column: modified_date
  batch_size: 10000

transformation:
  rules:
    - name: data_quality
      type: validation
      enabled: true
    - name: deduplication
      type: distinct
      enabled: true

loading:
  mode: merge
  conflict_resolution: update

monitoring:
  alerts:
    - type: failure
      channel: email
    - type: sla_breach
      threshold_minutes: 60
'''


# =============================================================================
# ETL AGENT ORCHESTRATOR
# =============================================================================

@dataclass
class ETLWorkflowConfig:
    """Configuration for the ETL agent workflow"""
    github_token: Optional[str] = None
    github_repo: Optional[str] = None
    branch_prefix: str = "etl/agent"
    create_draft_pr: bool = False
    
    @classmethod
    def from_env(cls) -> "ETLWorkflowConfig":
        return cls(
            github_token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT"),
            github_repo=os.environ.get("GITHUB_REPO"),
        )


class ETLAgentOrchestrator:
    """
    Main orchestrator for ETL Agent workflows.
    Manages the flow between specialized ETL agents and GitHub operations.
    """
    
    def __init__(self, config: Optional[ETLWorkflowConfig] = None, llm_service=None):
        self.config = config or ETLWorkflowConfig.from_env()
        self.llm_service = llm_service
        self.github_client: Optional[GitHubClient] = None
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all ETL agents"""
        self.schema_agent = SchemaExtractionAgent(self.llm_service)
        self.sql_agent = SQLGenerationAgent(self.llm_service)
        self.pipeline_agent = PipelineCreationAgent(self.llm_service)
        
        self.agent_registry = {
            ETLTaskType.SCHEMA_EXTRACTION: self.schema_agent,
            ETLTaskType.SQL_GENERATION: self.sql_agent,
            ETLTaskType.PIPELINE_CREATION: self.pipeline_agent,
        }
        
        logger.info("ETL Agent Orchestrator initialized with all agents")
    
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
    
    def determine_task_type(self, context: ETLContext) -> ETLTaskType:
        """Determine the appropriate task type based on context"""
        # Check source files for hints
        if context.source_files:
            extensions = [f.split(".")[-1] for f in context.source_files if "." in f]
            if any(ext in ["sql", "ddl"] for ext in extensions):
                return ETLTaskType.SQL_GENERATION
            if any(ext in ["yaml", "yml", "json"] for ext in extensions):
                return ETLTaskType.PIPELINE_CREATION
        
        # Check config hints
        if context.config.get("generate_sql"):
            return ETLTaskType.SQL_GENERATION
        if context.config.get("create_pipeline"):
            return ETLTaskType.PIPELINE_CREATION
        if context.config.get("extract_schema"):
            return ETLTaskType.SCHEMA_EXTRACTION
        
        return ETLTaskType.SQL_GENERATION  # Default
    
    def process_etl_task(self, context: ETLContext) -> Dict[str, Any]:
        """
        Process an ETL task through the complete agent pipeline
        
        Args:
            context: ETLContext with task details
            
        Returns:
            Dict: Complete processing results
        """
        try:
            logger.info(f"Starting ETL processing for task: {context.task_id}")
            
            # Determine task type if unknown
            if context.task_type == ETLTaskType.UNKNOWN:
                context.task_type = self.determine_task_type(context)
            
            # Get appropriate agent
            agent = self.agent_registry.get(context.task_type)
            if not agent:
                return self._create_error_response(
                    context.task_id, "No agent available", 
                    f"No agent for task type: {context.task_type.value}"
                )
            
            # Execute agent
            response = agent.execute(context)
            
            return {
                "task_id": context.task_id,
                "processing_status": "success" if response.success else "failed",
                "task_type": response.task_type.value,
                "result": response.result,
                "generated_files": response.generated_files,
                "reasoning": response.reasoning,
                "confidence_score": response.confidence_score,
                "next_actions": response.next_actions,
                "warnings": response.warnings,
                "timestamp": response.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing ETL task {context.task_id}: {str(e)}")
            return self._create_error_response(context.task_id, "Processing failed", str(e))
    
    async def execute_workflow(self, context: ETLContext) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Execute the complete ETL workflow with streaming events
        
        Args:
            context: ETLContext with task details
            
        Yields:
            WorkflowEvent objects for each step
        """
        try:
            # Step 1: Initialize
            yield WorkflowEvent(
                step=WorkflowStep.INIT,
                status="started",
                message="Initializing ETL workflow..."
            )
            
            self.github_client = self._init_github()
            repo_info = self.github_client.get_repo_info()
            default_branch = repo_info["default_branch"]
            
            yield WorkflowEvent(
                step=WorkflowStep.INIT,
                status="completed",
                message=f"Connected to {repo_info['full_name']}",
                data={"repo": repo_info["full_name"], "default_branch": default_branch}
            )
            
            # Step 2: Extract source information
            yield WorkflowEvent(
                step=WorkflowStep.EXTRACT,
                status="started",
                message="Extracting source information..."
            )
            
            if context.source_files:
                source_data = []
                for file_path in context.source_files[:5]:  # Limit for safety
                    try:
                        content = self.github_client.get_file_contents_decoded(file_path)
                        source_data.append({"path": file_path, "content": content[:1000]})
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")
                
                yield WorkflowEvent(
                    step=WorkflowStep.EXTRACT,
                    status="completed",
                    message=f"Extracted {len(source_data)} source files",
                    data={"files_extracted": len(source_data)}
                )
            else:
                yield WorkflowEvent(
                    step=WorkflowStep.EXTRACT,
                    status="completed",
                    message="No source files specified, using defaults"
                )
            
            # Step 3: Process through agent
            yield WorkflowEvent(
                step=WorkflowStep.GENERATE_CODE,
                status="started",
                message=f"Generating ETL code for {context.task_type.value}..."
            )
            
            result = self.process_etl_task(context)
            
            if result["processing_status"] != "success":
                yield WorkflowEvent(
                    step=WorkflowStep.ERROR,
                    status="error",
                    message=result.get("reasoning", "Processing failed"),
                    data=result
                )
                return
            
            yield WorkflowEvent(
                step=WorkflowStep.GENERATE_CODE,
                status="completed",
                message="Code generation completed",
                data={"files_generated": len(result.get("generated_files", {}))}
            )
            
            # Step 4: Create branch
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            branch_name = f"{self.config.branch_prefix}/{context.task_id.lower()}-{timestamp}"
            
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_BRANCH,
                status="started",
                message=f"Creating branch {branch_name}..."
            )
            
            self.github_client.create_branch(branch_name, default_branch)
            
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_BRANCH,
                status="completed",
                message=f"Created branch: {branch_name}",
                data={"branch": branch_name}
            )
            
            # Step 5: Commit files
            generated_files = result.get("generated_files", {})
            if generated_files:
                yield WorkflowEvent(
                    step=WorkflowStep.COMMIT_FILES,
                    status="started",
                    message=f"Committing {len(generated_files)} files..."
                )
                
                commit_message = f"""feat(etl): Add ETL code for {context.task_id}

Generated by GitHub ETL Agent
Task Type: {context.task_type.value}

Files:
{chr(10).join(f'- {f}' for f in generated_files.keys())}
"""
                
                commit = self.github_client.create_files_batch(
                    generated_files, commit_message, branch_name
                )
                
                yield WorkflowEvent(
                    step=WorkflowStep.COMMIT_FILES,
                    status="completed",
                    message=f"Committed {len(generated_files)} files",
                    data={"commit_sha": commit["sha"][:7], "files": list(generated_files.keys())}
                )
            
            # Step 6: Create PR
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_PR,
                status="started",
                message="Creating pull request..."
            )
            
            pr_title = f"[ETL] {context.task_id}: Auto-generated ETL code"
            pr_body = f"""## ETL Code Generation

**Task ID:** {context.task_id}
**Task Type:** {context.task_type.value}
**Generated by:** GitHub ETL Agent

### Files Created
{chr(10).join(f'- `{f}`' for f in generated_files.keys())}

### Summary
{result.get('reasoning', 'ETL code generated successfully')}

### Next Actions
{chr(10).join(f'- [ ] {a}' for a in result.get('next_actions', []))}

---
*This PR was automatically created by the GitHub ETL Agent*
"""
            
            pr = self.github_client.create_pull_request(
                title=pr_title, body=pr_body, head_branch=branch_name,
                base_branch=default_branch, draft=self.config.create_draft_pr
            )
            
            yield WorkflowEvent(
                step=WorkflowStep.CREATE_PR,
                status="completed",
                message=f"Created PR #{pr['number']}",
                data={"pr_number": pr["number"], "pr_url": pr["html_url"]}
            )
            
            # Complete
            yield WorkflowEvent(
                step=WorkflowStep.COMPLETE,
                status="completed",
                message="🎉 ETL workflow completed successfully!",
                data={
                    "branch": branch_name,
                    "pr_number": pr["number"],
                    "pr_url": pr["html_url"],
                    "files_created": list(generated_files.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            yield WorkflowEvent(
                step=WorkflowStep.ERROR,
                status="error",
                message=f"Workflow failed: {str(e)}",
                data={"error": str(e)}
            )
    
    def _create_error_response(self, task_id: str, error_type: str, details: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "task_id": task_id,
            "processing_status": "error",
            "error_type": error_type,
            "error_details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "orchestrator_status": "operational",
            "agents": {
                "schema_extraction": {"name": self.schema_agent.name, "type": self.schema_agent.task_type.value},
                "sql_generation": {"name": self.sql_agent.name, "type": self.sql_agent.task_type.value},
                "pipeline_creation": {"name": self.pipeline_agent.name, "type": self.pipeline_agent.task_type.value},
            },
            "config": {
                "github_repo": self.config.github_repo,
                "branch_prefix": self.config.branch_prefix
            }
        }


# =============================================================================
# API INTEGRATION (FastAPI)
# =============================================================================

def create_api_routes():
    """Create FastAPI routes for the ETL agent"""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    
    app = FastAPI(title="GitHub ETL Agent API", version="1.0.0")
    
    class ETLTaskRequest(BaseModel):
        task_id: str
        source_repo: str
        target_repo: Optional[str] = None
        task_type: Optional[str] = None
        source_files: Optional[List[str]] = None
        target_schema: Optional[str] = None
        config: Optional[Dict[str, Any]] = None
    
    @app.post("/api/etl/process")
    async def process_etl_task(request: ETLTaskRequest):
        """Process an ETL task"""
        try:
            task_type = ETLTaskType(request.task_type) if request.task_type else ETLTaskType.UNKNOWN
            
            context = ETLContext(
                task_id=request.task_id,
                source_repo=request.source_repo,
                target_repo=request.target_repo,
                task_type=task_type,
                task_complexity=TaskComplexity.MEDIUM,
                source_files=request.source_files or [],
                target_schema=request.target_schema,
                config=request.config or {}
            )
            
            orchestrator = ETLAgentOrchestrator()
            result = orchestrator.process_etl_task(context)
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/etl/workflow/stream")
    async def stream_etl_workflow(
        task_id: str = Query(...),
        source_repo: str = Query(...),
        task_type: str = Query(default="sql_generation"),
        source_files: str = Query(default="")
    ):
        """Stream ETL workflow execution via SSE"""
        
        async def event_generator():
            task_type_enum = ETLTaskType(task_type)
            files = [f.strip() for f in source_files.split(",") if f.strip()]
            
            context = ETLContext(
                task_id=task_id,
                source_repo=source_repo,
                target_repo=None,
                task_type=task_type_enum,
                task_complexity=TaskComplexity.MEDIUM,
                source_files=files
            )
            
            orchestrator = ETLAgentOrchestrator()
            async for event in orchestrator.execute_workflow(context):
                yield event.to_sse()
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    @app.get("/api/etl/status")
    async def get_status():
        """Get ETL agent status"""
        orchestrator = ETLAgentOrchestrator()
        return orchestrator.get_status()
    
    return app


# =============================================================================
# CLI / TEST RUNNER
# =============================================================================

def run_test():
    """Test the ETL Agent locally"""
    print("\n" + "="*60)
    print("🔧 GitHub ETL Agent - Test Run")
    print("="*60)
    
    # Create context
    context = ETLContext(
        task_id="ETL-001",
        source_repo="owner/source-repo",
        target_repo="owner/target-repo",
        task_type=ETLTaskType.SQL_GENERATION,
        task_complexity=TaskComplexity.MEDIUM,
        target_schema="dbo",
        config={"generate_sql": True}
    )
    
    # Initialize orchestrator (without GitHub for testing)
    orchestrator = ETLAgentOrchestrator()
    
    print("\n📋 Test Configuration:")
    print(f"   Task ID: {context.task_id}")
    print(f"   Task Type: {context.task_type.value}")
    print(f"   Source Repo: {context.source_repo}")
    
    # Process task
    print("\n" + "-"*60)
    print("🚀 Processing ETL Task...")
    print("-"*60)
    
    result = orchestrator.process_etl_task(context)
    
    print(f"\n✅ Status: {result['processing_status']}")
    print(f"📝 Reasoning: {result['reasoning']}")
    print(f"🎯 Confidence: {result['confidence_score']}")
    
    if result.get("generated_files"):
        print(f"\n📁 Generated Files:")
        for file_path in result["generated_files"].keys():
            print(f"   - {file_path}")
    
    if result.get("next_actions"):
        print(f"\n📌 Next Actions:")
        for action in result["next_actions"]:
            print(f"   - {action}")
    
    print("\n" + "="*60)
    print("✅ Test Complete!")
    print("="*60)
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_test()