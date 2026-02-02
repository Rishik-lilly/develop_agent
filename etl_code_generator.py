"""
ETL Code Generator
Generates SQL scripts, stored procedures, and ETL artifacts from JIRA tickets
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from enum import Enum
import requests

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SQLType(Enum):
    """Types of SQL artifacts"""
    DDL = "ddl"                    # CREATE TABLE, ALTER, etc.
    STORED_PROC = "stored_proc"   # Stored procedures
    INSERT = "insert"             # INSERT/MERGE statements
    EXTRACT = "extract"           # SELECT queries for extraction
    VALIDATION = "validation"     # Data quality queries


@dataclass
class ETLConfig:
    """ETL Generator configuration"""
    output_dir: str = "./generated_sql"
    
    # Azure OpenAI
    azure_endpoint: str = "https://openai-gis-sdlc-automation-instance.openai.azure.com/"
    azure_api_key: str = "81dcede85f694a0a8a4ab2874950b6cc"
    azure_deployment: str = "gpt-4o-mini"
    azure_api_version: str = "2024-12-01-preview"
    
    # SQL Settings
    database_type: str = "sqlserver"  # sqlserver, postgresql, snowflake, databricks


class ETLJiraParser:
    """Parse ETL-focused JIRA tickets"""
    
    def __init__(self, dump_path: str):
        self.dump_path = dump_path
        with open(dump_path, 'r') as f:
            self.data = json.load(f)
    
    def get_issues(self) -> List[Dict[str, Any]]:
        return self.data.get("issues", [])
    
    def get_issue_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        for issue in self.get_issues():
            if issue.get("key") == key:
                return issue
        return None
    
    def detect_sql_type(self, issue: Dict[str, Any]) -> SQLType:
        """Detect what type of SQL artifact to generate"""
        labels = [l.lower() for l in issue.get("labels", [])]
        summary = issue.get("summary", "").lower()
        
        if "ddl" in labels or "create" in summary or "dimension" in labels or "fact-table" in labels:
            return SQLType.DDL
        elif "stored-procedure" in labels or "procedure" in summary or "scd2" in labels:
            return SQLType.STORED_PROC
        elif "extract" in labels or "extract" in summary:
            return SQLType.EXTRACT
        elif "validation" in labels or "data-quality" in labels or "quality" in summary:
            return SQLType.VALIDATION
        elif "insert" in labels or "load" in summary or "fact-load" in labels:
            return SQLType.INSERT
        else:
            return SQLType.STORED_PROC  # Default to stored proc


class ETLPromptBuilder:
    """Build prompts for ETL SQL generation"""
    
    @staticmethod
    def build_ddl_prompt(issue: Dict[str, Any]) -> str:
        """Build prompt for DDL (CREATE TABLE) generation"""
        ac = "\n".join("  - " + c for c in issue.get("acceptance_criteria", []))
        
        return """You are an expert SQL developer specializing in data warehouses and ETL.
Generate a complete, production-ready SQL DDL script for the following requirement.

## JIRA Ticket: {key}
**Summary:** {summary}

**Description:**
{description}

**Target Schema:** {schema}
**Target Table:** {table}

**Acceptance Criteria:**
{ac}

## Requirements:
1. Use proper data types (VARCHAR, INT, DECIMAL, DATETIME2, etc.)
2. Include PRIMARY KEY constraint
3. Add appropriate NOT NULL constraints
4. Include DEFAULT values where sensible
5. Add comments for documentation
6. Follow naming conventions (snake_case)
7. Include CREATE INDEX statements for common query patterns

Generate ONLY the SQL code. Start with a header comment block.
Use T-SQL syntax for SQL Server.""".format(
            key=issue.get('key'),
            summary=issue.get('summary'),
            description=issue.get('description', ''),
            schema=issue.get('target_schema', 'dbo'),
            table=issue.get('target_table', 'TABLE_NAME'),
            ac=ac
        )
    
    @staticmethod
    def build_stored_proc_prompt(issue: Dict[str, Any]) -> str:
        """Build prompt for stored procedure generation"""
        ac = "\n".join("  - " + c for c in issue.get("acceptance_criteria", []))
        
        lookups = issue.get("dimension_lookups", [])
        lookup_info = ""
        if lookups:
            lookup_info = "\n**Dimension Lookups:**\n"
            for l in lookups:
                lookup_info += f"  - {l.get('dim_table')}: lookup {l.get('surrogate_key')} using {l.get('business_key')}\n"
        
        return """You are an expert SQL developer specializing in ETL stored procedures.
Generate a complete, production-ready stored procedure for the following requirement.

## JIRA Ticket: {key}
**Summary:** {summary}

**Description:**
{description}

**Source Table:** {source_schema}.{source_table}
**Target Table:** {target_schema}.{target_table}
**Business Key:** {business_key}
{lookup_info}
**Acceptance Criteria:**
{ac}

## Requirements:
1. Use CREATE OR ALTER PROCEDURE syntax
2. Include proper parameter declarations
3. Add TRY-CATCH error handling
4. Include transaction management (BEGIN TRAN, COMMIT, ROLLBACK)
5. Log row counts and execution metrics
6. Add comments explaining the logic
7. Use MERGE statement for upserts where appropriate
8. Handle SCD Type 2 logic if mentioned
9. Include SET NOCOUNT ON

Generate ONLY the SQL code. Start with a header comment block.
Use T-SQL syntax for SQL Server.""".format(
            key=issue.get('key'),
            summary=issue.get('summary'),
            description=issue.get('description', ''),
            source_schema=issue.get('source_schema', 'staging'),
            source_table=issue.get('source_table', 'SOURCE_TABLE'),
            target_schema=issue.get('target_schema', 'dbo'),
            target_table=issue.get('target_table', 'TARGET_TABLE'),
            business_key=issue.get('business_key', 'id'),
            lookup_info=lookup_info,
            ac=ac
        )
    
    @staticmethod
    def build_insert_prompt(issue: Dict[str, Any]) -> str:
        """Build prompt for INSERT/MERGE statement generation"""
        ac = "\n".join("  - " + c for c in issue.get("acceptance_criteria", []))
        
        return """You are an expert SQL developer specializing in ETL data loading.
Generate a complete INSERT or MERGE statement for the following requirement.

## JIRA Ticket: {key}
**Summary:** {summary}

**Description:**
{description}

**Source:** {source_schema}.{source_table}
**Target:** {target_schema}.{target_table}

**Acceptance Criteria:**
{ac}

## Requirements:
1. Use explicit column lists (no SELECT *)
2. Include proper JOIN conditions for dimension lookups
3. Handle NULL values with COALESCE/ISNULL
4. Add comments explaining transformations
5. Use MERGE for upsert patterns
6. Include OUTPUT clause to capture affected rows

Generate ONLY the SQL code. Use T-SQL syntax.""".format(
            key=issue.get('key'),
            summary=issue.get('summary'),
            description=issue.get('description', ''),
            source_schema=issue.get('source_schema', 'staging'),
            source_table=issue.get('source_table', 'SOURCE'),
            target_schema=issue.get('target_schema', 'dbo'),
            target_table=issue.get('target_table', 'TARGET'),
            ac=ac
        )
    
    @staticmethod
    def build_validation_prompt(issue: Dict[str, Any]) -> str:
        """Build prompt for data quality validation queries"""
        ac = "\n".join("  - " + c for c in issue.get("acceptance_criteria", []))
        tables = ", ".join(issue.get("tables_to_validate", []))
        
        return """You are an expert SQL developer specializing in data quality validation.
Generate a complete set of data quality check queries for the following requirement.

## JIRA Ticket: {key}
**Summary:** {summary}

**Description:**
{description}

**Tables to Validate:** {tables}
**Schema:** {schema}

**Acceptance Criteria:**
{ac}

## Requirements:
1. Each check should be a separate named query
2. Include NULL checks for required columns
3. Include duplicate checks on business keys
4. Include referential integrity checks
5. Include range/value validation checks
6. Output should include: check_name, table_name, status (PASS/FAIL), record_count, details
7. Wrap in a stored procedure that logs results to a DQ_RESULTS table
8. Include summary statistics

Generate ONLY the SQL code. Use T-SQL syntax.""".format(
            key=issue.get('key'),
            summary=issue.get('summary'),
            description=issue.get('description', ''),
            tables=tables,
            schema=issue.get('target_schema', 'dbo'),
            ac=ac
        )
    
    @staticmethod
    def build_extract_prompt(issue: Dict[str, Any]) -> str:
        """Build prompt for extraction query generation"""
        ac = "\n".join("  - " + c for c in issue.get("acceptance_criteria", []))
        
        return """You are an expert SQL developer specializing in ETL extraction.
Generate a parameterized extraction query for the following requirement.

## JIRA Ticket: {key}
**Summary:** {summary}

**Description:**
{description}

**Source Table:** {source_schema}.{source_table}
**Watermark Column:** {watermark}

**Acceptance Criteria:**
{ac}

## Requirements:
1. Use explicit column list
2. Parameterize the watermark date (@LastExtractDate)
3. Include proper WHERE clause for incremental logic
4. Add ORDER BY for consistent processing
5. Include comments explaining the extraction logic
6. Handle timezone if applicable
7. Consider adding NOLOCK hint for read performance

Generate ONLY the SQL code. Use T-SQL syntax.""".format(
            key=issue.get('key'),
            summary=issue.get('summary'),
            description=issue.get('description', ''),
            source_schema=issue.get('source_schema', 'dbo'),
            source_table=issue.get('source_table', 'SOURCE'),
            watermark=issue.get('watermark_column', 'modified_date'),
            ac=ac
        )


class AzureOpenAIGenerator:
    """Generate SQL using Azure OpenAI"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
    
    def generate(self, prompt: str) -> Generator[str, None, None]:
        """Generate SQL with streaming"""
        url = f"{self.config.azure_endpoint}openai/deployments/{self.config.azure_deployment}/chat/completions?api-version={self.config.azure_api_version}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.azure_api_key
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert SQL developer. Generate clean, production-ready SQL code. Output ONLY SQL code with comments, no markdown code blocks."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.3,  # Lower for more consistent SQL
            "stream": True
        }
        
        response = requests.post(url, headers=headers, json=payload, stream=True, verify=False, timeout=60)
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
                            content = choices[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue


class ETLCodeGenerator:
    """Main ETL code generator"""
    
    def __init__(self, config: Optional[ETLConfig] = None):
        self.config = config or ETLConfig()
        self.llm = AzureOpenAIGenerator(self.config)
        self.prompt_builder = ETLPromptBuilder()
    
    def generate_from_issue(self, issue: Dict[str, Any], sql_type: SQLType) -> Generator[str, None, None]:
        """Generate SQL from a JIRA issue"""
        
        # Build appropriate prompt
        if sql_type == SQLType.DDL:
            prompt = self.prompt_builder.build_ddl_prompt(issue)
        elif sql_type == SQLType.STORED_PROC:
            prompt = self.prompt_builder.build_stored_proc_prompt(issue)
        elif sql_type == SQLType.INSERT:
            prompt = self.prompt_builder.build_insert_prompt(issue)
        elif sql_type == SQLType.VALIDATION:
            prompt = self.prompt_builder.build_validation_prompt(issue)
        elif sql_type == SQLType.EXTRACT:
            prompt = self.prompt_builder.build_extract_prompt(issue)
        else:
            prompt = self.prompt_builder.build_stored_proc_prompt(issue)
        
        # Generate with LLM
        yield from self.llm.generate(prompt)
    
    def generate_and_save(self, issue: Dict[str, Any], sql_type: SQLType) -> str:
        """Generate SQL and save to file"""
        
        # Generate
        sql_code = ""
        print(f"\n🔄 Generating {sql_type.value} for {issue.get('key')}...")
        print("-" * 50)
        
        for chunk in self.generate_from_issue(issue, sql_type):
            sql_code += chunk
            print(chunk, end="", flush=True)
        
        print("\n" + "-" * 50)
        
        # Save to file
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        key = issue.get("key", "UNKNOWN").lower().replace("-", "_")
        table = issue.get("target_table", "table").lower()
        
        filename_map = {
            SQLType.DDL: f"{key}_ddl_{table}.sql",
            SQLType.STORED_PROC: f"{key}_sp_{table}.sql",
            SQLType.INSERT: f"{key}_insert_{table}.sql",
            SQLType.VALIDATION: f"{key}_dq_validation.sql",
            SQLType.EXTRACT: f"{key}_extract.sql"
        }
        
        filename = filename_map.get(sql_type, f"{key}.sql")
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(sql_code)
        
        print(f"\n✅ Saved to: {filepath}")
        return sql_code


# ============================================================================
# Demo / Test
# ============================================================================

def demo():
    """Demo ETL code generation"""
    print("\n" + "="*60)
    print("🔧 ETL Code Generator Demo")
    print("="*60)
    
    # Load ETL JIRA dump
    parser = ETLJiraParser("sample_etl_jira_dump.json")
    issues = parser.get_issues()
    
    print(f"\n📋 Found {len(issues)} ETL issues:")
    for issue in issues:
        sql_type = parser.detect_sql_type(issue)
        print(f"   • {issue['key']}: {issue['summary'][:40]}... [{sql_type.value}]")
    
    # Generate for one issue
    print("\n" + "="*60)
    print("🚀 Generating SQL for ETL-101 (Dimension Table DDL)")
    print("="*60)
    
    config = ETLConfig()
    generator = ETLCodeGenerator(config)
    
    issue = parser.get_issue_by_key("ETL-101")
    sql_type = parser.detect_sql_type(issue)
    
    generator.generate_and_save(issue, sql_type)
    
    print("\n✅ Demo complete! Check ./generated_sql/ folder")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        key = sys.argv[1]
        parser = ETLJiraParser("sample_etl_jira_dump.json")
        issue = parser.get_issue_by_key(key)
        
        if issue:
            config = ETLConfig()
            generator = ETLCodeGenerator(config)
            sql_type = parser.detect_sql_type(issue)
            generator.generate_and_save(issue, sql_type)
        else:
            print(f"Issue {key} not found")
    else:
        demo()