#!/usr/bin/env python
"""
Build script for TurboQuant C++ extension.
Handles compilation, testing, and fallback.
Run: python build_turboquant.py
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description):
    """Execute shell command and return success status."""
    print(f"\n{'='*70}")
    print(f"📦 {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║       TurboQuant C++ Extension Build Script                      ║
    ║  Compiles vector quantization for 4x faster embeddings          ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    
    # Step 1: Install dependencies
    print("📋 Step 1: Installing build dependencies...")
    if not run_command(
        f"{sys.executable} -m pip install -q pybind11 numpy",
        "Installing pybind11 and numpy"
    ):
        print("⚠️  Warning: Could not install pybind11")
    
    # Step 2: Try setup.py build (Windows/Linux)
    print("\n📋 Step 2: Building with setup.py...")
    if run_command(
        f"{sys.executable} setup_turboquant.py build_ext --inplace",
        "Building TurboQuant extension"
    ):
        print("✅ Successfully built turboquant_pybind extension!")
        
        # Test import
        print("\n📋 Step 3: Testing import...")
        try:
            import turboquant_pybind
            print("✅ turboquant_pybind imported successfully!")
            print(f"   - Location: {turboquant_pybind.__file__}")
            return 0
        except ImportError as e:
            print(f"❌ Import failed: {e}")
    
    # Step 3: Try CMake build
    if platform.system() != "Windows":
        print("\n📋 Attempting CMake build...")
        if run_command("cmake --version > /dev/null 2>&1", "Checking CMake"):
            if run_command(
                "cmake -B build && cmake --build build --config Release",
                "Building with CMake"
            ):
                print("✅ CMake build successful!")
                return 0
    
    # Fallback
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    ⚠️  Build Failed - Fallback Mode              ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    The C++ quantization extension could not be built, but CODI will
    continue to work with full-dimensional embeddings.
    
    To enable quantization:
    1. Install Visual Studio Build Tools (Windows) or gcc (Linux)
    2. Run: {sys.executable} setup_turboquant.py build_ext --inplace
    3. Or temporarily disable: set CODI_USE_QUANTIZATION=false
    
    Performance will be reduced but functionality remains intact.
    """)
    
    # Test fallback
    print("\n📋 Testing fallback (QuantizedEmbeddings without C++)...")
    try:
        from quantized_embeddings import QuantizedEmbeddings
        e = QuantizedEmbeddings(use_quantization=False)
        stats = e.get_cache_stats()
        print(f"✅ Fallback working: {stats}")
        print("   Using unquantized embeddings (slower but functional)")
        return 0
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
