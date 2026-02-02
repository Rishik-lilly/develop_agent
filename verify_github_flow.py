#!/usr/bin/env python3
"""
GitHub Flow Verification Script
Tests the complete workflow: Branch -> Issue -> Commit -> PR
Uses mock code generation (no LLM needed)
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
    
    print("\n" + "="*60)
    print("🔧 GitHub Flow Verification")
    print("="*60)
    
    # Try environment variables first
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    repo = os.environ.get("GITHUB_REPO")
    
    # If not set, ask user
    if not token:
        print("\n⚠️  GITHUB_TOKEN not found in environment")
        token = input("   Enter your GitHub token: ").strip()
    else:
        print(f"\n✅ GITHUB_TOKEN: {token[:10]}...")
    
    if not repo:
        print("\n⚠️  GITHUB_REPO not found in environment")
        repo = input("   Enter repository (owner/repo): ").strip()
    else:
        print(f"✅ GITHUB_REPO: {repo}")
    
    if not token or not repo:
        print("\n❌ Missing required configuration")
        sys.exit(1)
    
    return token, repo

# ============================================================================
# Main Verification Flow
# ============================================================================

def run_verification():
    """Run the complete verification flow"""
    
    token, repo = get_config()
    
    # Import our modules
    from github_client import GitHubClient, GitHubConfig
    from jira_to_code import MockCodeGenerator, Config, GenerationMode, JiraParser
    
    # Setup
    owner, repo_name = repo.split("/")
    config = GitHubConfig(token=token, owner=owner, repo=repo_name)
    client = GitHubClient(config)
    
    # Test data
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    test_jira_key = f"TEST-{timestamp[-6:]}"
    branch_name = f"test/verify-agent-{timestamp}"
    
    # Track created resources for cleanup
    created_resources = {
        "branch": None,
        "issue": None,
        "pr": None
    }
    
    print("\n" + "-"*60)
    print("📋 Test Configuration")
    print("-"*60)
    print(f"   Repository: {repo}")
    print(f"   Test JIRA Key: {test_jira_key}")
    print(f"   Branch Name: {branch_name}")
    
    try:
        # Step 1: Test Connection
        print("\n" + "-"*60)
        print("STEP 1: Testing GitHub Connection")
        print("-"*60)
        
        repo_info = client.get_repo_info()
        default_branch = repo_info["default_branch"]
        
        print(f"   ✅ Connected to: {repo_info['full_name']}")
        print(f"   ✅ Default branch: {default_branch}")
        print(f"   ✅ Private: {repo_info['private']}")
        
        # Step 2: Create Branch
        print("\n" + "-"*60)
        print("STEP 2: Creating Feature Branch")
        print("-"*60)
        
        branch_ref = client.create_branch(branch_name, default_branch)
        created_resources["branch"] = branch_name
        
        print(f"   ✅ Created branch: {branch_name}")
        print(f"   ✅ Branch URL: https://github.com/{repo}/tree/{branch_name}")
        
        # Step 3: Create Issue
        print("\n" + "-"*60)
        print("STEP 3: Creating GitHub Issue")
        print("-"*60)
        
        issue_title = f"[{test_jira_key}] Test: Verify Agent Workflow"
        issue_body = f"""## Test Issue for Agent Verification

This issue was created automatically to verify the agent workflow.

### Details
- **JIRA Key:** {test_jira_key}
- **Created:** {datetime.now(timezone.utc).isoformat()}
- **Branch:** `{branch_name}`

### Purpose
Testing the complete flow:
1. ✅ Create branch
2. ✅ Create issue
3. ⏳ Generate code
4. ⏳ Commit files
5. ⏳ Create PR

---
*This is a test issue. Safe to close/delete.*
"""
        
        issue = client.create_issue(
            title=issue_title,
            body=issue_body,
            labels=["test", "automated"]
        )
        created_resources["issue"] = issue["number"]
        
        print(f"   ✅ Created issue #{issue['number']}")
        print(f"   ✅ Issue URL: {issue['html_url']}")
        
        # Step 4: Generate Mock Code
        print("\n" + "-"*60)
        print("STEP 4: Generating Mock Code")
        print("-"*60)
        
        # Create a test JIRA issue
        test_issue = {
            "key": test_jira_key,
            "type": "Story",
            "summary": "Test User Service API",
            "description": "Test API for verification purposes",
            "acceptance_criteria": [
                "GET /api/test returns success",
                "POST /api/test creates resource"
            ],
            "labels": ["test", "api"],
            "tech_stack": ["Python", "FastAPI"],
            "priority": "Low"
        }
        
        # Generate code
        gen_config = Config(mode=GenerationMode.MOCK)
        generator = MockCodeGenerator(gen_config)
        
        # Detect code type
        parser = JiraParser.__new__(JiraParser)
        parser.data = {"issues": [test_issue]}
        code_type = parser.detect_code_type(test_issue)
        
        code = ""
        for chunk in generator.generate(test_issue, code_type):
            code += chunk
        
        print(f"   ✅ Generated {len(code.splitlines())} lines of {code_type} code")
        
        # Step 5: Commit Files
        print("\n" + "-"*60)
        print("STEP 5: Committing Generated Code")
        print("-"*60)
        
        # Determine file path
        file_ext = "tsx" if code_type == "frontend" else "py"
        file_path = f"src/generated/{test_jira_key.lower().replace('-', '_')}.{file_ext}"
        
        commit_message = f"""feat({test_jira_key}): Add generated test code

This commit was created automatically by the Agent verification script.

Closes #{issue['number']}
"""
        
        commit = client.create_files_batch(
            files={file_path: code},
            message=commit_message,
            branch=branch_name
        )
        
        print(f"   ✅ Created file: {file_path}")
        print(f"   ✅ Commit SHA: {commit['sha'][:7]}")
        print(f"   ✅ Commit URL: https://github.com/{repo}/commit/{commit['sha']}")
        
        # Step 6: Create Pull Request
        print("\n" + "-"*60)
        print("STEP 6: Creating Pull Request")
        print("-"*60)
        
        pr_title = f"[{test_jira_key}] Test: Verify Agent Workflow"
        pr_body = f"""## Summary
Automated test PR to verify the agent workflow.

Closes #{issue['number']}

## Changes
- Generated mock code for testing

## Files
- `{file_path}`

## Verification Checklist
- [x] Branch created
- [x] Issue created  
- [x] Code generated
- [x] Files committed
- [x] PR created

---
*This is a test PR. Safe to close without merging.*
"""
        
        pr = client.create_pull_request(
            title=pr_title,
            body=pr_body,
            head_branch=branch_name,
            base_branch=default_branch
        )
        created_resources["pr"] = pr["number"]
        
        print(f"   ✅ Created PR #{pr['number']}")
        print(f"   ✅ PR URL: {pr['html_url']}")
        
        # Summary
        print("\n" + "="*60)
        print("🎉 VERIFICATION COMPLETE!")
        print("="*60)
        print("\n📊 Created Resources:")
        print(f"   • Branch: https://github.com/{repo}/tree/{branch_name}")
        print(f"   • Issue:  {issue['html_url']}")
        print(f"   • PR:     {pr['html_url']}")
        print(f"   • File:   https://github.com/{repo}/blob/{branch_name}/{file_path}")
        
        print("\n✅ All steps completed successfully!")
        print("\n" + "-"*60)
        
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
        print("\n🧹 Attempting cleanup of partial resources...")
        cleanup_resources(client, repo, created_resources)
        raise

def cleanup_resources(client, repo, resources):
    """Clean up created test resources"""
    
    print("\n" + "-"*60)
    print("🧹 Cleaning Up Test Resources")
    print("-"*60)
    
    # Close PR (if created)
    if resources.get("pr"):
        try:
            # Close PR by updating its state
            client._request(
                "PATCH",
                f"{client.repo_path}/pulls/{resources['pr']}",
                json={"state": "closed"}
            )
            print(f"   ✅ Closed PR #{resources['pr']}")
        except Exception as e:
            print(f"   ⚠️  Could not close PR: {e}")
    
    # Close Issue (if created)
    if resources.get("issue"):
        try:
            client.close_issue(resources["issue"])
            print(f"   ✅ Closed Issue #{resources['issue']}")
        except Exception as e:
            print(f"   ⚠️  Could not close issue: {e}")
    
    # Delete Branch (if created)
    if resources.get("branch"):
        try:
            client.delete_branch(resources["branch"])
            print(f"   ✅ Deleted branch: {resources['branch']}")
        except Exception as e:
            print(f"   ⚠️  Could not delete branch: {e}")
    
    print("\n✅ Cleanup complete!")

# ============================================================================
# Quick Test Mode
# ============================================================================

def quick_test():
    """Quick connection test without creating resources"""
    
    print("\n" + "="*60)
    print("🔍 Quick GitHub Connection Test")
    print("="*60)
    
    token, repo = get_config()
    
    from github_client import GitHubClient, GitHubConfig
    
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
        
        permissions = repo_info.get("permissions", {})
        print(f"\n🔐 Permissions:")
        print(f"   Admin: {permissions.get('admin', 'N/A')}")
        print(f"   Push: {permissions.get('push', 'N/A')}")
        print(f"   Pull: {permissions.get('pull', 'N/A')}")
        
        if permissions.get("push"):
            print("\n✅ You have push access - full workflow will work!")
        else:
            print("\n⚠️  No push access - you may not be able to create branches/commits")
            
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🤖 Agent GitHub Flow Verification")
    print("="*60)
    
    print("\nChoose an option:")
    print("  1. Quick test (connection only, no changes)")
    print("  2. Full verification (creates branch, issue, PR)")
    print("  3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        print("\n⚠️  This will create REAL resources in your GitHub repo!")
        confirm = input("   Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            run_verification()
        else:
            print("   Cancelled.")
    elif choice == "3":
        print("\nBye!")
    else:
        print("\nInvalid choice.")

if __name__ == "__main__":
    # Allow direct mode via command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_test()
        elif sys.argv[1] == "--full":
            run_verification()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python verify_github_flow.py [--quick|--full]")
    else:
        main()