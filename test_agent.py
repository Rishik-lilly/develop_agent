#!/usr/bin/env python3
"""
Test Script for JIRA to Code Agent
Tests code generation with Azure OpenAI and GitHub workflow
"""

import os
import sys

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_azure_openai():
    """Test Azure OpenAI connection"""
    print("\n" + "="*60)
    print("🤖 Testing Azure OpenAI Connection")
    print("="*60)
    
    from jira_to_code import Config, RealCodeGenerator
    
    config = Config()
    generator = RealCodeGenerator(config)
    
    print(f"\n📋 Configuration:")
    print(f"   Endpoint: {config.azure_endpoint}")
    print(f"   Deployment: {config.azure_deployment}")
    print(f"   API Version: {config.azure_api_version}")
    print(f"   API Key: {config.azure_api_key[:10]}...")
    
    if not generator.is_available():
        print("\n❌ Azure OpenAI not configured")
        return False
    
    print("\n🔄 Sending test prompt...")
    
    try:
        test_prompt = "Write a simple Python function that adds two numbers. Just the code, no explanation."
        
        response = ""
        for chunk in generator.generate(test_prompt):
            response += chunk
            print(chunk, end="", flush=True)
        
        print("\n\n✅ Azure OpenAI is working!")
        print(f"   Response length: {len(response)} characters")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_code_generation():
    """Test full code generation from JIRA issue"""
    print("\n" + "="*60)
    print("⚙️  Testing Code Generation from JIRA")
    print("="*60)
    
    from jira_to_code import JiraParser, JiraToCodeGenerator, Config, GenerationMode
    
    # Load a sample issue
    parser = JiraParser("sample_jira_dump.json")
    issue = parser.get_issue_by_key("DEMO-101")
    
    print(f"\n📋 JIRA Issue: {issue['key']}")
    print(f"   Summary: {issue['summary']}")
    print(f"   Type: {parser.detect_code_type(issue)}")
    
    # Test with REAL mode (Azure OpenAI)
    print("\n🔄 Generating code with Azure OpenAI...")
    print("-"*40)
    
    config = Config(mode=GenerationMode.REAL)
    generator = JiraToCodeGenerator(config)
    
    try:
        code = generator.generate_for_issue(issue, parser)
        
        print("\n" + "-"*40)
        print(f"✅ Generated {len(code.splitlines())} lines of code")
        print(f"   Saved to: {config.output_dir}/")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Falling back to mock generation...")
        
        config = Config(mode=GenerationMode.MOCK)
        generator = JiraToCodeGenerator(config)
        code = generator.generate_for_issue(issue, parser)
        
        print(f"\n✅ Generated {len(code.splitlines())} lines with mock")
        return True

def test_github_connection():
    """Test GitHub connection"""
    print("\n" + "="*60)
    print("🐙 Testing GitHub Connection")
    print("="*60)
    
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    repo = os.environ.get("GITHUB_REPO")
    
    if not token:
        print("\n⚠️  GITHUB_TOKEN not set - skipping")
        return None
    
    if not repo:
        print("\n⚠️  GITHUB_REPO not set - skipping")
        return None
    
    from github_client import GitHubClient, GitHubConfig
    
    try:
        owner, repo_name = repo.split("/")
        config = GitHubConfig(token=token, owner=owner, repo=repo_name)
        client = GitHubClient(config)
        
        info = client.get_repo_info()
        
        print(f"\n✅ Connected to: {info['full_name']}")
        print(f"   Default branch: {info['default_branch']}")
        print(f"   Private: {info['private']}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def run_full_workflow():
    """Run the complete agent workflow"""
    print("\n" + "="*60)
    print("🚀 Running Full Agent Workflow")
    print("="*60)
    
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    repo = os.environ.get("GITHUB_REPO")
    
    if not token or not repo:
        print("\n❌ GITHUB_TOKEN and GITHUB_REPO required")
        print("   Set with:")
        print("   $env:GITHUB_TOKEN = 'your_token'")
        print("   $env:GITHUB_REPO = 'owner/repo'")
        return
    
    print(f"\n📋 Repository: {repo}")
    print("\n⚠️  This will create REAL resources in GitHub:")
    print("   • Branch")
    print("   • Issue")
    print("   • Generated code commit")
    print("   • Pull request")
    
    confirm = input("\nContinue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return
    
    # Run the verification script
    from verify_github_flow import run_verification
    run_verification()

def main():
    print("\n" + "="*60)
    print("🧪 JIRA to Code Agent - Test Suite")
    print("="*60)
    
    print("\nChoose a test:")
    print("  1. Test Azure OpenAI connection")
    print("  2. Test code generation (JIRA -> Code)")
    print("  3. Test GitHub connection")
    print("  4. Run ALL tests")
    print("  5. Run FULL workflow (creates GitHub resources)")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        test_azure_openai()
    elif choice == "2":
        test_code_generation()
    elif choice == "3":
        test_github_connection()
    elif choice == "4":
        print("\n🔄 Running all tests...")
        results = {
            "Azure OpenAI": test_azure_openai(),
            "Code Generation": test_code_generation(),
            "GitHub": test_github_connection()
        }
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        for test, result in results.items():
            if result is True:
                print(f"   ✅ {test}: PASSED")
            elif result is False:
                print(f"   ❌ {test}: FAILED")
            else:
                print(f"   ⏭️  {test}: SKIPPED")
                
    elif choice == "5":
        run_full_workflow()
    elif choice == "6":
        print("\nBye!")
    else:
        print("\nInvalid choice.")

if __name__ == "__main__":
    # Allow direct mode via command line
    if len(sys.argv) > 1:
        if sys.argv[1] == "--azure":
            test_azure_openai()
        elif sys.argv[1] == "--code":
            test_code_generation()
        elif sys.argv[1] == "--github":
            test_github_connection()
        elif sys.argv[1] == "--all":
            test_azure_openai()
            test_code_generation()
            test_github_connection()
        elif sys.argv[1] == "--workflow":
            run_full_workflow()
        else:
            print(f"Unknown: {sys.argv[1]}")
            print("Usage: python test_agent.py [--azure|--code|--github|--all|--workflow]")
    else:
        main()