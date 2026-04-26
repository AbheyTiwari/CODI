#!/usr/bin/env python3
"""
Quick test script to verify content diversity fix works correctly.
"""
import os
import sys
import tempfile
import shutil

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Setup environment
os.environ.setdefault("CODI_WORKING_DIR", os.getcwd())
os.environ.setdefault("CODI_CHROMA_DIR", os.path.join(_REPO_ROOT, "chroma_db", "test"))

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from agent import create_agent
from state.temp_db import RunState

def test_content_diversity():
    """Test if creating multiple files produces different content."""
    print("\n" + "="*80)
    print("TESTING: Content Diversity Fix")
    print("="*80 + "\n")
    
    # Create a test directory
    test_dir = tempfile.mkdtemp()
    os.environ["CODI_WORKING_DIR"] = test_dir
    
    try:
        agent = create_agent()
        
        # Test request: create two different files
        user_input = (
            "create portfolio.html with a projects section showing 3 sample projects "
            "and a skills section. also create simple.html as a basic starter page "
            "with just a heading and button"
        )
        
        print(f"📝 Test Input:\n{user_input}\n")
        print("🚀 Running agent...\n")
        
        result = agent.invoke({
            "input": user_input,
            "history": ""
        })
        
        output = result.get("output", "")
        tool_outputs = result.get("tool_outputs", [])
        
        print(f"✅ Agent Output:\n{output}\n")
        print(f"📊 Tool Outputs ({len(tool_outputs)} items):\n")
        for i, output in enumerate(tool_outputs[-5:], 1):
            if len(output) > 300:
                print(f"  {i}. {output[:300]}...\n")
            else:
                print(f"  {i}. {output}\n")
        
        # Check if files were created
        print("\n" + "="*80)
        print("📁 Files Created:")
        print("="*80 + "\n")
        
        if os.path.exists(os.path.join(test_dir, "portfolio.html")):
            with open(os.path.join(test_dir, "portfolio.html")) as f:
                portfolio_content = f.read()
            print(f"✓ portfolio.html created ({len(portfolio_content)} chars)")
            print(f"  Content preview: {portfolio_content[:200]}...\n")
        else:
            print("✗ portfolio.html NOT found\n")
        
        if os.path.exists(os.path.join(test_dir, "simple.html")):
            with open(os.path.join(test_dir, "simple.html")) as f:
                simple_content = f.read()
            print(f"✓ simple.html created ({len(simple_content)} chars)")
            print(f"  Content preview: {simple_content[:200]}...\n")
        else:
            print("✗ simple.html NOT found\n")
        
        # Check content diversity
        print("="*80)
        print("✨ Diversity Check:")
        print("="*80 + "\n")
        
        if os.path.exists(os.path.join(test_dir, "portfolio.html")) and \
           os.path.exists(os.path.join(test_dir, "simple.html")):
            with open(os.path.join(test_dir, "portfolio.html")) as f:
                portfolio = f.read()
            with open(os.path.join(test_dir, "simple.html")) as f:
                simple = f.read()
            
            if portfolio == simple:
                print("❌ FAILED: Files are IDENTICAL!")
                print(f"   Both have {len(portfolio)} characters")
            else:
                # Check if portfolio has portfolio-specific content
                portfolio_keywords = ["project", "skill", "experience", "work"]
                simple_keywords = ["simple", "starter", "basic"]
                
                portfolio_matches = sum(1 for kw in portfolio_keywords if kw.lower() in portfolio.lower())
                simple_matches = sum(1 for kw in simple_keywords if kw.lower() in simple.lower())
                
                print(f"📊 Content Analysis:")
                print(f"   Portfolio file matches portfolio keywords: {portfolio_matches}/4")
                print(f"   Simple file matches simple keywords: {simple_matches}/3")
                
                if portfolio_matches > 0:
                    print(f"\n✅ SUCCESS: Files have DIFFERENT CONTENT!")
                    print(f"   Portfolio: {len(portfolio)} chars, Simple: {len(simple)} chars")
                    print(f"   Difference: {abs(len(portfolio) - len(simple))} chars")
                else:
                    print(f"\n⚠️  WARNING: Content may still be generic")
                    print(f"   Portfolio doesn't have portfolio-specific keywords")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test directory: {test_dir}\n")

if __name__ == "__main__":
    test_content_diversity()
