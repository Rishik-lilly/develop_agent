#!/usr/bin/env python3
"""
Quick Demo Runner for JIRA to Code Generator
Run this to test the generation without any setup
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jira_to_code import JiraToCodeGenerator, Config, GenerationMode

def demo_single_issue():
    """Demo: Generate code for a single issue"""
    print("\n" + "="*70)
    print("🎯 DEMO: Generating code for a single JIRA issue")
    print("="*70)
    
    config = Config(mode=GenerationMode.MOCK)  # Use mock for quick demo
    generator = JiraToCodeGenerator(config)
    
    results = generator.generate_from_dump(
        "sample_jira_dump.json",
        issue_keys=["DEMO-101"]  # Just the auth API
    )
    
    return results

def demo_all_issues():
    """Demo: Generate code for all issues in dump"""
    print("\n" + "="*70)
    print("🚀 DEMO: Generating code for ALL JIRA issues")
    print("="*70)
    
    config = Config(mode=GenerationMode.MOCK)
    generator = JiraToCodeGenerator(config)
    
    results = generator.generate_from_dump("sample_jira_dump.json")
    
    return results

def demo_with_real_llm():
    """Demo: Use real LLM (requires API key)"""
    print("\n" + "="*70)
    print("🤖 DEMO: Using REAL LLM for code generation")
    print("="*70)
    
    api_key = os.environ.get("CORTEX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ No API key found!")
        print("   Set CORTEX_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("   Falling back to mock generation...")
        return demo_single_issue()
    
    config = Config(
        mode=GenerationMode.REAL,
        api_key=api_key
    )
    generator = JiraToCodeGenerator(config)
    
    results = generator.generate_from_dump(
        "sample_jira_dump.json",
        issue_keys=["DEMO-102"]  # Frontend component
    )
    
    return results

def demo_auto_mode():
    """Demo: Auto mode - tries real, falls back to mock"""
    print("\n" + "="*70)
    print("⚡ DEMO: AUTO mode (real LLM with mock fallback)")
    print("="*70)
    
    config = Config(mode=GenerationMode.AUTO)
    generator = JiraToCodeGenerator(config)
    
    results = generator.generate_from_dump(
        "sample_jira_dump.json",
        issue_keys=["DEMO-103"]  # Search service
    )
    
    return results

def interactive_menu():
    """Interactive menu for demos"""
    print("\n" + "="*70)
    print("        🔧 JIRA TO CODE GENERATOR - DEMO")
    print("="*70)
    print("\nChoose a demo to run:\n")
    print("  1. Generate code for ONE issue (mock - fast)")
    print("  2. Generate code for ALL issues (mock - shows variety)")
    print("  3. Generate with REAL LLM (requires API key)")
    print("  4. AUTO mode (tries real, falls back to mock)")
    print("  5. Exit")
    print()
    
    while True:
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            demo_single_issue()
        elif choice == "2":
            demo_all_issues()
        elif choice == "3":
            demo_with_real_llm()
        elif choice == "4":
            demo_auto_mode()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")
            continue
        
        print("\n" + "-"*50)
        input("Press Enter to continue...")
        print("\n")

if __name__ == "__main__":
    # Check if dump file exists
    if not os.path.exists("sample_jira_dump.json"):
        print("❌ Error: sample_jira_dump.json not found!")
        print("   Make sure you're in the correct directory")
        sys.exit(1)
    
    # Run interactive menu or direct demo
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--single":
            demo_single_issue()
        elif arg == "--all":
            demo_all_issues()
        elif arg == "--real":
            demo_with_real_llm()
        elif arg == "--auto":
            demo_auto_mode()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python run_demo.py [--single|--all|--real|--auto]")
    else:
        interactive_menu()
