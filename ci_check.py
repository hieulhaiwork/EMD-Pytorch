#!/usr/bin/env python3
"""
Simple CI compatibility check - no CUDA required
"""

import sys
import platform
import os

def check_python_version():
    """Check Python version compatibility"""
    print("Python Version Check")
    version = sys.version_info
    print(f"  Current: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 7):
        print("  ✅ Compatible")
        return True
    else:
        print("  ❌ Requires Python 3.7+")
        return False

def check_platform():
    """Check platform compatibility"""
    print("\nPlatform Check")
    system = platform.system()
    print(f"  System: {system}")
    print(f"  Architecture: {platform.machine()}")
    
    if system in ['Windows', 'Linux', 'Darwin']:
        print("  ✅ Supported platform")
        return True
    else:
        print("  ⚠️  Untested platform")
        return False

def check_basic_imports():
    """Check if basic dependencies can be imported"""
    print("\nBasic Import Check")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print("  ✅ PyTorch import successful")
    except ImportError:
        print("  ❌ PyTorch not available")
        return False
    
    try:
        import numpy
        print(f"  NumPy version: {numpy.__version__}")
        print("  ✅ NumPy import successful")
    except ImportError:
        print("  ❌ NumPy not available")
        return False
    
    return True

def check_package_structure():
    """Check if package structure is correct"""
    print("\nPackage Structure Check")
    
    # Check if emd directory exists
    if not os.path.exists('emd'):
        print("  ❌ emd directory not found")
        return False
    print("  ✅ emd directory exists")
    
    # Check if __init__.py exists
    if not os.path.exists('emd/__init__.py'):
        print("  ❌ emd/__init__.py not found")
        return False
    print("  ✅ emd/__init__.py exists")
    
    # Check if emd.py exists
    if not os.path.exists('emd/emd.py'):
        print("  ❌ emd/emd.py not found")
        return False
    print("  ✅ emd/emd.py exists")
    
    # Check if CUDA directory exists
    if not os.path.exists('emd/cuda'):
        print("  ❌ emd/cuda directory not found")
        return False
    print("  ✅ emd/cuda directory exists")
    
    return True

def main():
    """Run all compatibility checks"""
    print("=" * 50)
    print("EMD PyTorch CI Compatibility Check")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_platform(),
        check_basic_imports(),
        check_package_structure()
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 All compatibility checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()