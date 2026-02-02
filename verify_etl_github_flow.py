#!/usr/bin/env python3
"""
GitHub ETL Flow Verification Script
====================================
Tests the complete ETL workflow: Branch -> Generate Code -> Commit -> PR
Validates all GitHub operations and ETL agent functionality.

Usage:
    python verify_etl_github_flow.py
    
Environment Variables:
    GITHUB_TOKEN or GITHUB_PAT - GitHub personal access token
    GITHUB_REPO - Repository in format "owner/repo"
"""

import os
import sys
from datetime import datetime, timezone

# Suppress SSL warnings for corporate environments
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
# Configuration
# ============================================================================

def get_config():
    """Get GitHub configuration from environment or user input"""
    
    print("\n" + "="*70)
    print("🔧 GitHub ETL Flow Verification")
    print("="*70)
    
    # Try environment variables first
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    repo = os.environ.get("GITHUB_REPO")
    
    # If not set, ask user
    if not token:
        print("\n⚠️  GITHUB_TOKEN not found in environment")
        token = input("   Enter your GitHub token: ").strip()
    else:
        print(f"\n✅ GITHUB_TOKEN: {token[:10]}...{token[-4:]}")
    
    if not repo:
        print("\n⚠️  GITHUB_REPO not found in environment")
        repo = input("   Enter repository (owner/repo): ").strip()
    else:
        print(f"✅ GITHUB_REPO: {repo}")
    
    if not token or not repo:
        print("\n❌ Missing required configuration")
        sys.exit(1)
    
    return token, repo


def print_step(step_num: int, title: str):
    """Print formatted step header"""
    print("\n" + "-"*70)
    print(f"STEP {step_num}: {title}")
    print("-"*70)


def print_success(message: str):
    """Print success message"""
    print(f"   ✅ {message}")


def print_warning(message: str):
    """Print warning message"""
    print(f"   ⚠️  {message}")


def print_error(message: str):
    """Print error message"""
    print(f"   ❌ {message}")


def print_info(message: str):
    """Print info message"""
    print(f"   ℹ️  {message}")


# ============================================================================
# Import ETL Agent Components
# ============================================================================

def import_etl_components():
    """Import ETL agent components"""
    try:
        from github_client import (
            GitHubClient, GitHubConfig)
        from etl_agent_orchestrator import (
            ETLAgentOrchestrator, ETLWorkflowConfig,
            ETLContext, ETLTaskType, TaskComplexity,
            SQLGenerationAgent, SchemaExtractionAgent, PipelineCreationAgent
        )
        return True
    except ImportError as e:
        print(f"\n⚠️  Could not import from github_etl_agent: {e}")
        print("   Using inline implementation...")
        return False


# ============================================================================
# Inline GitHub Client (fallback if module not available)
# ============================================================================

import base64
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class GitHubConfig:
    """GitHub configuration"""
    token: str
    owner: str
    repo: str
    base_url: str = "https://api.github.com"
    proxy: Optional[str] = None
    verify_ssl: bool = False


class GitHubClient:
    """GitHub API client for ETL operations"""
    
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
        return self._request("GET", self.repo_path)
    
    def get_default_branch(self) -> str:
        return self.get_repo_info()["default_branch"]
    
    def get_branch_sha(self, branch_name: str) -> str:
        branch = self._request("GET", f"{self.repo_path}/branches/{branch_name}")
        return branch["commit"]["sha"]
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> Dict[str, Any]:
        if from_branch is None:
            from_branch = self.get_default_branch()
        sha = self.get_branch_sha(from_branch)
        return self._request(
            "POST", f"{self.repo_path}/git/refs",
            json={"ref": f"refs/heads/{branch_name}", "sha": sha}
        )
    
    def delete_branch(self, branch_name: str) -> None:
        self._request("DELETE", f"{self.repo_path}/git/refs/heads/{branch_name}")
    
    def create_issue(self, title: str, body: str, labels: List[str] = None) -> Dict[str, Any]:
        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        return self._request("POST", f"{self.repo_path}/issues", json=data)
    
    def close_issue(self, issue_number: int) -> Dict[str, Any]:
        return self._request(
            "PATCH", f"{self.repo_path}/issues/{issue_number}",
            json={"state": "closed"}
        )
    
    def create_files_batch(self, files: Dict[str, str], message: str, branch: str) -> Dict[str, Any]:
        branch_sha = self.get_branch_sha(branch)
        commit = self._request("GET", f"{self.repo_path}/git/commits/{branch_sha}")
        base_tree = commit["tree"]["sha"]
        
        tree_items = []
        for path, content in files.items():
            blob = self._request(
                "POST", f"{self.repo_path}/git/blobs",
                json={"content": content, "encoding": "utf-8"}
            )
            tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        
        new_tree = self._request(
            "POST", f"{self.repo_path}/git/trees",
            json={"base_tree": base_tree, "tree": tree_items}
        )
        
        new_commit = self._request(
            "POST", f"{self.repo_path}/git/commits",
            json={"message": message, "tree": new_tree["sha"], "parents": [branch_sha]}
        )
        
        self._request(
            "PATCH", f"{self.repo_path}/git/refs/heads/{branch}",
            json={"sha": new_commit["sha"]}
        )
        
        return new_commit
    
    def create_pull_request(
        self, title: str, body: str, head_branch: str,
        base_branch: Optional[str] = None, draft: bool = False
    ) -> Dict[str, Any]:
        if base_branch is None:
            base_branch = self.get_default_branch()
        return self._request(
            "POST", f"{self.repo_path}/pulls",
            json={"title": title, "body": body, "head": head_branch, "base": base_branch, "draft": draft}
        )


# ============================================================================
# Mock ETL Code Generator
# ============================================================================

class MockETLCodeGenerator:
    """Generate mock ETL code for testing"""
    
    @staticmethod
    def generate_stored_procedure(task_id: str, schema: str = "dbo") -> str:
        """Generate a test stored procedure"""
        return f'''-- ============================================================
-- ETL Stored Procedure: {task_id}
-- Generated by: GitHub ETL Agent Verification Script
-- Generated: {datetime.now(timezone.utc).isoformat()}
-- ============================================================

CREATE OR ALTER PROCEDURE [{schema}].[usp_ETL_{task_id.replace("-", "_")}]
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
        -- EXTRACT Phase
        -- ========================================
        IF @DebugMode = 1
            PRINT 'Starting extraction phase...';
        
        -- Extract source data into staging
        SELECT *
        INTO #StagingData
        FROM SourceTable
        WHERE ModifiedDate >= DATEADD(DAY, -1, @BatchDate);
        
        SET @RowsAffected = @@ROWCOUNT;
        
        IF @DebugMode = 1
            PRINT 'Extracted ' + CAST(@RowsAffected AS VARCHAR(20)) + ' rows';
        
        -- ========================================
        -- TRANSFORM Phase
        -- ========================================
        IF @DebugMode = 1
            PRINT 'Starting transformation phase...';
        
        -- Apply business rules and transformations
        UPDATE #StagingData
        SET 
            ProcessedFlag = 1,
            ProcessedDate = GETDATE();
        
        -- ========================================
        -- LOAD Phase
        -- ========================================
        IF @DebugMode = 1
            PRINT 'Starting load phase...';
        
        -- Merge into target table
        MERGE [{schema}].[TargetTable] AS target
        USING #StagingData AS source
        ON target.BusinessKey = source.BusinessKey
        
        WHEN MATCHED THEN
            UPDATE SET
                target.Column1 = source.Column1,
                target.ModifiedDate = GETDATE()
        
        WHEN NOT MATCHED THEN
            INSERT (BusinessKey, Column1, CreatedDate)
            VALUES (source.BusinessKey, source.Column1, GETDATE());
        
        SET @RowsAffected = @@ROWCOUNT;
        
        COMMIT TRANSACTION;
        
        -- Log success
        IF @DebugMode = 1
        BEGIN
            PRINT 'ETL completed successfully.';
            PRINT 'Rows affected: ' + CAST(@RowsAffected AS VARCHAR(20));
            PRINT 'Duration: ' + CAST(DATEDIFF(SECOND, @StartTime, SYSDATETIME()) AS VARCHAR(20)) + ' seconds';
        END
        
        -- Return success
        SELECT 
            'SUCCESS' AS Status,
            @RowsAffected AS RowsAffected,
            DATEDIFF(SECOND, @StartTime, SYSDATETIME()) AS DurationSeconds;
        
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;
        
        SET @ErrorMessage = ERROR_MESSAGE();
        
        -- Log error
        INSERT INTO [{schema}].[ETL_ErrorLog] (ProcedureName, ErrorMessage, ErrorDate)
        VALUES ('usp_ETL_{task_id.replace("-", "_")}', @ErrorMessage, GETDATE());
        
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END
GO

-- Grant execute permission
GRANT EXECUTE ON [{schema}].[usp_ETL_{task_id.replace("-", "_")}] TO [ETL_Executor];
GO
'''
    
    @staticmethod
    def generate_pipeline_config(task_id: str) -> str:
        """Generate a test pipeline configuration"""
        return f'''# ============================================================
# ETL Pipeline Configuration: {task_id}
# Generated by: GitHub ETL Agent Verification Script
# Generated: {datetime.now(timezone.utc).isoformat()}
# ============================================================

pipeline:
  name: {task_id.lower().replace("-", "_")}_pipeline
  version: "1.0.0"
  description: "ETL pipeline for {task_id}"
  owner: etl_team
  tags:
    - automated
    - verification
    - {task_id}

schedule:
  cron: "0 2 * * *"  # Daily at 2 AM
  timezone: "UTC"
  catchup: false

source:
  type: database
  connection_id: source_db_connection
  schema: source_schema
  tables:
    - name: source_table_1
      incremental_column: modified_date
    - name: source_table_2
      incremental_column: updated_at

target:
  type: database
  connection_id: target_db_connection
  schema: target_schema

extraction:
  mode: incremental
  batch_size: 10000
  parallel_threads: 4
  retry_attempts: 3
  retry_delay_seconds: 60

transformation:
  steps:
    - name: data_quality_check
      type: validation
      rules:
        - column: id
          check: not_null
        - column: email
          check: valid_email
    
    - name: deduplication
      type: distinct
      key_columns:
        - business_key
    
    - name: standardization
      type: transform
      operations:
        - column: name
          operation: uppercase
        - column: date_field
          operation: to_date
          format: "YYYY-MM-DD"

loading:
  mode: merge
  merge_keys:
    - business_key
  conflict_resolution: update
  soft_delete: true
  soft_delete_column: is_active

monitoring:
  enabled: true
  alerts:
    - type: failure
      channels:
        - email
        - slack
    - type: sla_breach
      threshold_minutes: 60
      channels:
        - pagerduty
  
  metrics:
    - rows_extracted
    - rows_transformed
    - rows_loaded
    - duration_seconds
    - error_count

logging:
  level: INFO
  format: json
  destination: cloudwatch
'''
    
    @staticmethod
    def generate_airflow_dag(task_id: str) -> str:
        """Generate a test Airflow DAG"""
        task_name = task_id.lower().replace("-", "_")
        return f'''"""
ETL Pipeline DAG: {task_id}
Generated by: GitHub ETL Agent Verification Script
Generated: {datetime.now(timezone.utc).isoformat()}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.microsoft.mssql.operators.mssql import MsSqlOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

# Default arguments for the DAG
default_args = {{
    'owner': 'etl_agent',
    'depends_on_past': False,
    'email': ['etl-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}}

# DAG definition
dag = DAG(
    dag_id='{task_name}_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline generated by GitHub ETL Agent - {task_id}',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'automated', 'verification', '{task_id}'],
    doc_md="""
    ## {task_id} ETL Pipeline
    
    This DAG was automatically generated by the GitHub ETL Agent.
    
    ### Pipeline Steps:
    1. **Extract**: Pull data from source systems
    2. **Transform**: Apply business rules and data quality checks
    3. **Load**: Merge data into target tables
    
    ### Monitoring:
    - Alerts on failure via email
    - SLA: 60 minutes
    """,
)


def log_start(**context):
    """Log pipeline start"""
    print(f"Starting ETL pipeline: {task_id}")
    print(f"Execution date: {{context['execution_date']}}")
    return {{"status": "started", "timestamp": datetime.now().isoformat()}}


def extract_data(**context):
    """Extract data from source systems"""
    print("Executing extraction phase...")
    # TODO: Implement actual extraction logic
    rows_extracted = 1000  # Mock value
    context['ti'].xcom_push(key='rows_extracted', value=rows_extracted)
    return {{"rows_extracted": rows_extracted}}


def validate_data(**context):
    """Validate extracted data"""
    print("Executing validation phase...")
    rows_extracted = context['ti'].xcom_pull(key='rows_extracted', task_ids='extract')
    # TODO: Implement actual validation logic
    validation_passed = True
    if not validation_passed:
        raise ValueError("Data validation failed")
    return {{"validation_passed": validation_passed}}


def transform_data(**context):
    """Transform data according to business rules"""
    print("Executing transformation phase...")
    # TODO: Implement actual transformation logic
    rows_transformed = 950  # Mock value (some rows filtered out)
    context['ti'].xcom_push(key='rows_transformed', value=rows_transformed)
    return {{"rows_transformed": rows_transformed}}


def load_data(**context):
    """Load data into target systems"""
    print("Executing load phase...")
    rows_transformed = context['ti'].xcom_pull(key='rows_transformed', task_ids='transform')
    # TODO: Implement actual load logic
    rows_loaded = rows_transformed
    return {{"rows_loaded": rows_loaded}}


def log_completion(**context):
    """Log pipeline completion"""
    print(f"ETL pipeline completed: {task_id}")
    return {{"status": "completed", "timestamp": datetime.now().isoformat()}}


# Task definitions
with dag:
    
    start = DummyOperator(task_id='start')
    
    log_start_task = PythonOperator(
        task_id='log_start',
        python_callable=log_start,
        provide_context=True,
    )
    
    with TaskGroup(group_id='extraction') as extraction_group:
        extract_task = PythonOperator(
            task_id='extract',
            python_callable=extract_data,
            provide_context=True,
        )
        
        validate_task = PythonOperator(
            task_id='validate',
            python_callable=validate_data,
            provide_context=True,
        )
        
        extract_task >> validate_task
    
    with TaskGroup(group_id='transformation') as transformation_group:
        transform_task = PythonOperator(
            task_id='transform',
            python_callable=transform_data,
            provide_context=True,
        )
    
    with TaskGroup(group_id='loading') as loading_group:
        load_task = PythonOperator(
            task_id='load',
            python_callable=load_data,
            provide_context=True,
        )
        
        # Execute stored procedure
        execute_sp = MsSqlOperator(
            task_id='execute_stored_procedure',
            mssql_conn_id='mssql_default',
            sql="EXEC dbo.usp_ETL_{task_name} @BatchDate = '{{{{ ds }}}}', @DebugMode = 0",
        )
        
        load_task >> execute_sp
    
    log_complete_task = PythonOperator(
        task_id='log_completion',
        python_callable=log_completion,
        provide_context=True,
    )
    
    end = DummyOperator(task_id='end')
    
    # Define task dependencies
    start >> log_start_task >> extraction_group >> transformation_group >> loading_group >> log_complete_task >> end
'''


# ============================================================================
# Main Verification Flow
# ============================================================================

def run_verification():
    """Run the complete ETL verification flow"""
    
    token, repo = get_config()
    
    # Setup GitHub client
    owner, repo_name = repo.split("/")
    config = GitHubConfig(token=token, owner=owner, repo=repo_name)
    client = GitHubClient(config)
    
    # Test data
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    test_task_id = f"ETL-TEST-{timestamp[-6:]}"
    branch_name = f"test/etl-agent-verify-{timestamp}"
    
    # Track created resources for cleanup
    created_resources = {
        "branch": None,
        "issue": None,
        "pr": None
    }
    
    print("\n" + "-"*70)
    print("📋 Test Configuration")
    print("-"*70)
    print(f"   Repository: {repo}")
    print(f"   Test Task ID: {test_task_id}")
    print(f"   Branch Name: {branch_name}")
    
    try:
        # ================================================================
        # STEP 1: Test GitHub Connection
        # ================================================================
        print_step(1, "Testing GitHub Connection")
        
        repo_info = client.get_repo_info()
        default_branch = repo_info["default_branch"]
        
        print_success(f"Connected to: {repo_info['full_name']}")
        print_success(f"Default branch: {default_branch}")
        print_success(f"Private: {repo_info['private']}")
        print_success(f"Permissions: push={repo_info.get('permissions', {}).get('push', 'unknown')}")
        
        # ================================================================
        # STEP 2: Create Feature Branch
        # ================================================================
        print_step(2, "Creating Feature Branch")
        
        branch_ref = client.create_branch(branch_name, default_branch)
        created_resources["branch"] = branch_name
        
        print_success(f"Created branch: {branch_name}")
        print_success(f"Branch URL: https://github.com/{repo}/tree/{branch_name}")
        
        # ================================================================
        # STEP 3: Create GitHub Issue
        # ================================================================
        print_step(3, "Creating GitHub Issue")
        
        issue_title = f"[{test_task_id}] ETL Agent Verification Test"
        issue_body = f"""## ETL Agent Verification Test

This issue was created automatically to verify the GitHub ETL Agent workflow.

### Test Details
- **Task ID:** {test_task_id}
- **Created:** {datetime.now(timezone.utc).isoformat()}
- **Branch:** `{branch_name}`

### Verification Checklist
- [x] GitHub connection established
- [x] Branch created
- [x] Issue created
- [ ] ETL code generated
- [ ] Files committed
- [ ] PR created

### Generated Files
The following files will be generated and committed:
- `etl/stored_procedures/sp_{test_task_id.lower().replace("-", "_")}.sql`
- `etl/config/{test_task_id.lower().replace("-", "_")}_config.yaml`
- `etl/dags/{test_task_id.lower().replace("-", "_")}_dag.py`

---
*This is a test issue created by the ETL Agent verification script. Safe to close/delete.*
"""
        
        issue = client.create_issue(
            title=issue_title,
            body=issue_body,
            labels=["etl", "automated", "test"]
        )
        created_resources["issue"] = issue["number"]
        
        print_success(f"Created Issue #{issue['number']}")
        print_success(f"Issue URL: {issue['html_url']}")
        
        # ================================================================
        # STEP 4: Generate ETL Code
        # ================================================================
        print_step(4, "Generating ETL Code")
        
        generator = MockETLCodeGenerator()
        
        # Generate all ETL files
        sp_code = generator.generate_stored_procedure(test_task_id)
        config_code = generator.generate_pipeline_config(test_task_id)
        dag_code = generator.generate_airflow_dag(test_task_id)
        
        task_name = test_task_id.lower().replace("-", "_")
        files_to_commit = {
            f"etl/stored_procedures/sp_{task_name}.sql": sp_code,
            f"etl/config/{task_name}_config.yaml": config_code,
            f"etl/dags/{task_name}_dag.py": dag_code,
        }
        
        print_success(f"Generated stored procedure: {len(sp_code)} characters")
        print_success(f"Generated pipeline config: {len(config_code)} characters")
        print_success(f"Generated Airflow DAG: {len(dag_code)} characters")
        print_info(f"Total files to commit: {len(files_to_commit)}")
        
        # ================================================================
        # STEP 5: Commit Files
        # ================================================================
        print_step(5, "Committing Generated Files")
        
        commit_message = f"""feat(etl): Add ETL code for {test_task_id}

This commit was created automatically by the ETL Agent verification script.

Generated files:
- Stored procedure for data processing
- Pipeline configuration (YAML)
- Airflow DAG for orchestration

Closes #{issue['number']}
"""
        
        commit = client.create_files_batch(
            files=files_to_commit,
            message=commit_message,
            branch=branch_name
        )
        
        print_success(f"Commit SHA: {commit['sha'][:7]}")
        print_success(f"Commit URL: https://github.com/{repo}/commit/{commit['sha']}")
        
        for file_path in files_to_commit.keys():
            print_success(f"Created: {file_path}")
        
        # ================================================================
        # STEP 6: Create Pull Request
        # ================================================================
        print_step(6, "Creating Pull Request")
        
        pr_title = f"[{test_task_id}] ETL Agent Verification - Auto-generated Code"
        pr_body = f"""## ETL Agent Verification

**Task ID:** {test_task_id}
**Generated by:** GitHub ETL Agent Verification Script

Closes #{issue['number']}

### Summary
This PR contains automatically generated ETL code to verify the GitHub ETL Agent workflow.

### Files Created

| File | Description |
|------|-------------|
| `etl/stored_procedures/sp_{task_name}.sql` | SQL Server stored procedure for ETL |
| `etl/config/{task_name}_config.yaml` | Pipeline configuration |
| `etl/dags/{task_name}_dag.py` | Airflow DAG for orchestration |

### Verification Checklist

- [x] Branch created from `{default_branch}`
- [x] GitHub issue created and linked
- [x] ETL stored procedure generated
- [x] Pipeline configuration generated
- [x] Airflow DAG generated
- [x] All files committed
- [x] PR created and linked to issue

### Generated Code Preview

<details>
<summary>Stored Procedure (click to expand)</summary>

```sql
{sp_code[:500]}...
```

</details>

<details>
<summary>Pipeline Config (click to expand)</summary>

```yaml
{config_code[:500]}...
```

</details>

---
*This is a test PR created by the ETL Agent verification script.*
*Safe to close without merging.*
"""
        
        pr = client.create_pull_request(
            title=pr_title,
            body=pr_body,
            head_branch=branch_name,
            base_branch=default_branch
        )
        created_resources["pr"] = pr["number"]
        
        print_success(f"Created PR #{pr['number']}")
        print_success(f"PR URL: {pr['html_url']}")
        
        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "="*70)
        print("🎉 VERIFICATION COMPLETE!")
        print("="*70)
        
        print("\n📊 Created Resources:")
        print(f"   • Branch: https://github.com/{repo}/tree/{branch_name}")
        print(f"   • Issue:  {issue['html_url']}")
        print(f"   • PR:     {pr['html_url']}")
        print(f"\n📁 Generated Files:")
        for file_path in files_to_commit.keys():
            print(f"   • https://github.com/{repo}/blob/{branch_name}/{file_path}")
        
        print("\n✅ All steps completed successfully!")
        print("\n" + "-"*70)
        
        # Cleanup option
        cleanup = input("\n🧹 Delete test resources? (yes/no): ").strip().lower()
        
        if cleanup == "yes":
            cleanup_resources(client, repo, created_resources)
        else:
            print("\n💡 To manually clean up later:")
            print(f"   1. Close PR #{pr['number']}")
            print(f"   2. Close Issue #{issue['number']}")
            print(f"   3. Delete branch: {branch_name}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🧹 Attempting cleanup of partial resources...")
        cleanup_resources(client, repo, created_resources)
        raise


def cleanup_resources(client: GitHubClient, repo: str, resources: dict):
    """Clean up created test resources"""
    
    print("\n" + "-"*70)
    print("🧹 Cleaning Up Test Resources")
    print("-"*70)
    
    # Close PR (if created)
    if resources.get("pr"):
        try:
            client._request(
                "PATCH",
                f"{client.repo_path}/pulls/{resources['pr']}",
                json={"state": "closed"}
            )
            print_success(f"Closed PR #{resources['pr']}")
        except Exception as e:
            print_warning(f"Could not close PR: {e}")
    
    # Close Issue (if created)
    if resources.get("issue"):
        try:
            client.close_issue(resources["issue"])
            print_success(f"Closed Issue #{resources['issue']}")
        except Exception as e:
            print_warning(f"Could not close issue: {e}")
    
    # Delete Branch (if created)
    if resources.get("branch"):
        try:
            client.delete_branch(resources["branch"])
            print_success(f"Deleted branch: {resources['branch']}")
        except Exception as e:
            print_warning(f"Could not delete branch: {e}")
    
    print("\n✅ Cleanup complete!")


# ============================================================================
# Quick Test Mode
# ============================================================================

def quick_test():
    """Quick connection test without creating resources"""
    
    print("\n" + "="*70)
    print("🔍 Quick GitHub Connection Test")
    print("="*70)
    
    token, repo = get_config()
    
    owner, repo_name = repo.split("/")
    config = GitHubConfig(token=token, owner=owner, repo=repo_name)
    client = GitHubClient(config)
    
    try:
        repo_info = client.get_repo_info()
        
        print(f"\n✅ Connection successful!")
        print(f"\n📊 Repository Info:")
        print(f"   Name: {repo_info['full_name']}")
        print(f"   Default Branch: {repo_info['default_branch']}")
        print(f"   Private: {repo_info['private']}")
        print(f"   Description: {repo_info.get('description', 'N/A')}")
        print(f"   Stars: {repo_info.get('stargazers_count', 0)}")
        print(f"   Forks: {repo_info.get('forks_count', 0)}")
        
        permissions = repo_info.get('permissions', {})
        print(f"\n🔐 Permissions:")
        print(f"   Admin: {permissions.get('admin', 'unknown')}")
        print(f"   Push: {permissions.get('push', 'unknown')}")
        print(f"   Pull: {permissions.get('pull', 'unknown')}")
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False
    
    return True


def test_etl_generation():
    """Test ETL code generation without GitHub"""
    
    print("\n" + "="*70)
    print("🧪 Test ETL Code Generation (No GitHub)")
    print("="*70)
    
    generator = MockETLCodeGenerator()
    test_task_id = "ETL-TEST-001"
    
    print(f"\n📋 Generating ETL code for: {test_task_id}")
    
    # Generate stored procedure
    sp_code = generator.generate_stored_procedure(test_task_id)
    print(f"\n✅ Stored Procedure: {len(sp_code)} characters")
    print("-"*50)
    print(sp_code[:800] + "\n...")
    
    # Generate config
    config_code = generator.generate_pipeline_config(test_task_id)
    print(f"\n✅ Pipeline Config: {len(config_code)} characters")
    print("-"*50)
    print(config_code[:600] + "\n...")
    
    # Generate DAG
    dag_code = generator.generate_airflow_dag(test_task_id)
    print(f"\n✅ Airflow DAG: {len(dag_code)} characters")
    print("-"*50)
    print(dag_code[:600] + "\n...")
    
    print("\n" + "="*70)
    print("✅ ETL Code Generation Test Complete!")
    print("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point with menu"""
    
    print("\n" + "="*70)
    print("🔧 GitHub ETL Agent - Verification Suite")
    print("="*70)
    
    print("\nChoose a test:")
    print("  1. Quick connection test (no resources created)")
    print("  2. Test ETL code generation (no GitHub)")
    print("  3. Run FULL verification (creates GitHub resources)")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        test_etl_generation()
    elif choice == "3":
        print("\n⚠️  This will create REAL resources in GitHub:")
        print("   • Branch")
        print("   • Issue")
        print("   • Generated ETL code files")
        print("   • Pull request")
        
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        if confirm == "yes":
            run_verification()
        else:
            print("Cancelled.")
    elif choice == "4":
        print("\nGoodbye!")
        sys.exit(0)
    else:
        print("\nInvalid choice. Please enter 1-4.")
        main()


if __name__ == "__main__":
    main()