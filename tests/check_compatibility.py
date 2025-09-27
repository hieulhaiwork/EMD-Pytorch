#!/usr/bin/env python3
"""
Platform compatibility checker for EMD CUDA extension
"""

import sys
import platform
import subprocess
import torch

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check")
    version = sys.version_info
    print(f"   Current: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 7):
        print("   ✅ Compatible")
        return True
    else:
        print("   ❌ Requires Python 3.7+")
        return False

def check_platform():
    """Check platform compatibility"""
    print("\n🖥️  Platform Check")
    system = platform.system()
    print(f"   System: {system}")
    print(f"   Architecture: {platform.machine()}")
    
    if system in ['Windows', 'Linux', 'Darwin']:
        print("   ✅ Supported platform")
        return True
    else:
        print("   ⚠️  Untested platform")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    print("\n🔥 PyTorch Check")
    try:
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print("   ✅ CUDA support available")
            return True
        else:
            print("   ❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"   ❌ PyTorch check failed: {e}")
        return False

def check_compiler():
    """Check compiler availability"""
    print("\n🔧 Compiler Check")
    system = platform.system()
    
    if system == "Windows":
        # Check for Visual Studio
        try:
            result = subprocess.run(['where', 'cl'], capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ Visual Studio compiler found")
                return True
            else:
                print("   ❌ Visual Studio compiler not found")
                print("   Install Visual Studio Build Tools")
                return False
        except:
            print("   ❌ Cannot check for Visual Studio compiler")
            return False
            
    elif system == "Linux":
        # Check for GCC
        try:
            result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"   ✅ GCC found: {version_line}")
                return True
            else:
                print("   ❌ GCC not found")
                return False
        except:
            print("   ❌ Cannot check for GCC")
            return False
            
    elif system == "Darwin":
        # Check for Clang (Xcode)
        try:
            result = subprocess.run(['clang', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"   ✅ Clang found: {version_line}")
                return True
            else:
                print("   ❌ Clang not found")
                print("   Install Xcode Command Line Tools")
                return False
        except:
            print("   ❌ Cannot check for Clang")
            return False
    
    return False

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    print("\n🚀 CUDA Toolkit Check")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_info = result.stdout
            print(f"   ✅ NVCC found:")
            for line in version_info.split('\n')[:3]:
                if line.strip():
                    print(f"      {line.strip()}")
            return True
        else:
            print("   ❌ NVCC not found")
            print("   Install CUDA Toolkit")
            return False
    except:
        print("   ❌ Cannot check CUDA toolkit")
        return False

def main():
    """Run all compatibility checks"""
    print("🔍 EMD CUDA Extension Compatibility Check")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_platform,
        check_pytorch,
        check_compiler,
        check_cuda_toolkit,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Summary")
    
    passed = sum(results)
    total = len(results)
    
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("   🎉 All checks passed! Your system is ready for EMD CUDA extension.")
        print("\n💡 Next steps:")
        print("   1. Run: python setup.py build_ext --inplace")
        print("   2. Test: python comprehensive_test.py")
    elif passed >= total - 1:
        print("   ⚠️  Most checks passed. You might encounter minor issues.")
        print("   🔧 Try building anyway, some issues might be non-critical.")
    else:
        print("   ❌ Multiple issues detected. Please resolve them before building.")
        print("\n🔧 Common solutions:")
        if not results[2]:  # PyTorch CUDA
            print("   - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        if not results[3]:  # Compiler
            if platform.system() == "Windows":
                print("   - Install Visual Studio Build Tools")
            elif platform.system() == "Linux":
                print("   - Install build tools: sudo apt-get install build-essential")
        if not results[4]:  # CUDA toolkit
            print("   - Install CUDA toolkit from NVIDIA website")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)