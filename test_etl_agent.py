#!/usr/bin/env python3
"""
ETL Agent Test Script
Test SQL generation for ETL pipelines
"""

import os
import sys

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from etl_code_generator import (
    ETLConfig, ETLCodeGenerator, ETLJiraParser, SQLType
)


def list_etl_issues():
    """List all ETL issues"""
    print("\n" + "="*60)
    print("📋 ETL JIRA Issues")
    print("="*60)
    
    parser = ETLJiraParser("sample_etl_jira_dump.json")
    issues = parser.get_issues()
    
    for issue in issues:
        sql_type = parser.detect_sql_type(issue)
        print(f"\n   {issue['key']}: {issue['summary']}")
        print(f"      Type: {sql_type.value}")
        print(f"      Labels: {', '.join(issue.get('labels', []))}")
        if issue.get('target_table'):
            print(f"      Target: {issue.get('target_schema', 'dbo')}.{issue['target_table']}")


def generate_single(key: str):
    """Generate SQL for a single issue"""
    print(f"\n🔄 Generating SQL for {key}...")
    
    parser = ETLJiraParser("sample_etl_jira_dump.json")
    issue = parser.get_issue_by_key(key)
    
    if not issue:
        print(f"❌ Issue {key} not found")
        return
    
    sql_type = parser.detect_sql_type(issue)
    config = ETLConfig()
    generator = ETLCodeGenerator(config)
    
    generator.generate_and_save(issue, sql_type)


def generate_all():
    """Generate SQL for all ETL issues"""
    print("\n" + "="*60)
    print("🚀 Generating ALL ETL SQL Scripts")
    print("="*60)
    
    parser = ETLJiraParser("sample_etl_jira_dump.json")
    issues = parser.get_issues()
    
    config = ETLConfig()
    generator = ETLCodeGenerator(config)
    
    for issue in issues:
        sql_type = parser.detect_sql_type(issue)
        print(f"\n{'='*60}")
        print(f"Processing: {issue['key']} ({sql_type.value})")
        print("="*60)
        
        try:
            generator.generate_and_save(issue, sql_type)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print(f"✅ Generated SQL for {len(issues)} issues")
    print(f"📁 Output: ./generated_sql/")
    print("="*60)


def generate_by_type(sql_type_str: str):
    """Generate all issues of a specific type"""
    type_map = {
        "ddl": SQLType.DDL,
        "proc": SQLType.STORED_PROC,
        "sp": SQLType.STORED_PROC,
        "insert": SQLType.INSERT,
        "extract": SQLType.EXTRACT,
        "validation": SQLType.VALIDATION,
        "dq": SQLType.VALIDATION
    }
    
    sql_type = type_map.get(sql_type_str.lower())
    if not sql_type:
        print(f"❌ Unknown type: {sql_type_str}")
        print(f"   Valid types: {', '.join(type_map.keys())}")
        return
    
    print(f"\n🔄 Generating all {sql_type.value} SQL...")
    
    parser = ETLJiraParser("sample_etl_jira_dump.json")
    issues = parser.get_issues()
    
    config = ETLConfig()
    generator = ETLCodeGenerator(config)
    
    count = 0
    for issue in issues:
        detected_type = parser.detect_sql_type(issue)
        if detected_type == sql_type:
            generator.generate_and_save(issue, sql_type)
            count += 1
    
    print(f"\n✅ Generated {count} {sql_type.value} scripts")


def interactive_menu():
    """Interactive menu"""
    print("\n" + "="*60)
    print("🔧 ETL SQL Generator")
    print("="*60)
    
    print("\nChoose an option:")
    print("  1. List all ETL issues")
    print("  2. Generate SQL for ONE issue")
    print("  3. Generate ALL SQL scripts")
    print("  4. Generate by type (DDL, StoredProc, etc.)")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        list_etl_issues()
    elif choice == "2":
        list_etl_issues()
        key = input("\nEnter issue key (e.g., ETL-101): ").strip().upper()
        generate_single(key)
    elif choice == "3":
        confirm = input("\nThis will generate ALL SQL scripts. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            generate_all()
    elif choice == "4":
        print("\nAvailable types: ddl, proc, insert, extract, validation")
        type_str = input("Enter type: ").strip()
        generate_by_type(type_str)
    elif choice == "5":
        print("\nBye!")
        return
    else:
        print("\nInvalid choice.")
    
    # Loop back
    input("\nPress Enter to continue...")
    interactive_menu()


def main():
    # Check if dump file exists
    if not os.path.exists("sample_etl_jira_dump.json"):
        print("❌ sample_etl_jira_dump.json not found!")
        return
    
    # Command line args
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--list":
            list_etl_issues()
        elif arg == "--all":
            generate_all()
        elif arg.startswith("--type="):
            type_str = arg.split("=")[1]
            generate_by_type(type_str)
        elif arg.startswith("ETL-"):
            generate_single(arg)
        else:
            print(f"Unknown argument: {arg}")
            print("\nUsage:")
            print("  python test_etl_agent.py --list          # List all issues")
            print("  python test_etl_agent.py --all           # Generate all SQL")
            print("  python test_etl_agent.py --type=ddl      # Generate by type")
            print("  python test_etl_agent.py ETL-101         # Generate single issue")
    else:
        interactive_menu()


if __name__ == "__main__":
    main()