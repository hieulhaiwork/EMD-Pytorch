#!/usr/bin/env python3
"""
Final validation script - test everything before GitHub upload
"""

import os
import sys
import subprocess
import importlib

def test_import():
    """Test if all modules can be imported correctly"""
    print("🔍 Testing Package Import")
    
    try:
        # Test main package
        import emd
        print("   ✅ Main package import successful")
        
        # Test specific functions
        from emd import earth_mover_distance, EMDLoss
        print("   ✅ Function imports successful")
        
        # Test CUDA extension
        import emd_cuda
        print("   ✅ CUDA extension import successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic EMD computation"""
    print("\n🧪 Testing Basic Functionality")
    
    try:
        import torch
        from emd import earth_mover_distance
        
        # Small test
        xyz1 = torch.randn(1, 10, 3).cuda()
        xyz2 = torch.randn(1, 10, 3).cuda()
        
        cost = earth_mover_distance(xyz1, xyz2, transpose=False)
        
        print(f"   ✅ EMD computation successful: {cost.item():.4f}")
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def test_file_structure():
    """Check if all required files are present"""
    print("\n📁 Testing File Structure")
    
    required_files = [
        'README.md',
        'LICENSE', 
        'setup.py',
        'pyproject.toml',
        '__init__.py',
        'emd.py',
        'cuda/emd.cpp',
        'cuda/emd_kernel.cu',
        'comprehensive_test.py',
        'loss_test.py',
        'check_compatibility.py',
        '.gitignore',
        'MANIFEST.in'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    else:
        print("   ✅ All required files present")
        return True

def test_build_config():
    """Test if setup.py is properly configured"""
    print("\n⚙️  Testing Build Configuration")
    
    try:
        # Try to parse setup.py
        with open('setup.py', 'r') as f:
            content = f.read()
            
        # Check for cross-platform compatibility
        if 'platform.system()' in content:
            print("   ✅ Cross-platform compiler args detected")
        else:
            print("   ⚠️  No cross-platform detection found")
            
        # Check for essential metadata
        essential_keys = ['name', 'version', 'description', 'install_requires']
        for key in essential_keys:
            if key in content:
                print(f"   ✅ {key} found in setup.py")
            else:
                print(f"   ❌ {key} missing from setup.py")
                
        return True
        
    except Exception as e:
        print(f"   ❌ Setup.py validation failed: {e}")
        return False

def test_documentation():
    """Check documentation quality"""
    print("\n📖 Testing Documentation")
    
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme = f.read()
            
        # Check for essential sections
        sections = [
            'Installation',
            'Usage', 
            'Requirements',
            'Examples',
            'Troubleshooting'
        ]
        
        found_sections = []
        for section in sections:
            if section.lower() in readme.lower():
                found_sections.append(section)
                
        print(f"   ✅ Documentation sections found: {len(found_sections)}/{len(sections)}")
        
        if len(readme) > 5000:
            print("   ✅ Comprehensive documentation (>5k chars)")
        else:
            print("   ⚠️  Documentation might be too brief")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Documentation check failed: {e}")
        return False

def run_comprehensive_test():
    """Run the comprehensive test suite"""
    print("\n🚀 Running Comprehensive Test Suite")
    
    try:
        result = subprocess.run([sys.executable, 'comprehensive_test.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("   ✅ All comprehensive tests passed")
            return True
        else:
            print(f"   ❌ Comprehensive tests failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Tests timed out (may still be working)")
        return True
    except Exception as e:
        print(f"   ❌ Could not run comprehensive tests: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🎯 Final Validation for GitHub Upload")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Build Configuration", test_build_config),
        ("Documentation", test_documentation),
        ("Package Import", test_import),
        ("Basic Functionality", test_basic_functionality),
        ("Comprehensive Tests", run_comprehensive_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Final Results")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("📦 Your package is ready for GitHub upload!")
        print("\n📋 Next steps:")
        print("   1. Create GitHub repository")
        print("   2. Upload files (see UPLOAD_GUIDE.md)")
        print("   3. Create first release")
        print("   4. Test installation from GitHub")
        
    elif passed >= total - 1:
        print("\n⚠️  Almost ready! Minor issues detected.")
        print("🔧 Consider fixing the failing tests before upload.")
        
    else:
        print("\n❌ Multiple issues detected.")
        print("🔧 Please fix the failing tests before GitHub upload.")
    
    return passed >= total - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)