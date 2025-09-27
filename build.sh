#!/bin/bash
# build.sh - Build script for Linux/macOS

set -e

echo "🚀 Building EMD CUDA extension for Linux/macOS..."

# Check if CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -f *.so

# Build the extension
echo "🔨 Building CUDA extension..."
python setup.py build_ext --inplace

# Check if build was successful
if [ -f *.so ]; then
    echo "✅ Build successful! CUDA extension created."
    echo ""
    echo "You can now test the module with:"
    echo "  python comprehensive_test.py"
else
    echo "❌ Build failed! No .so file found."
    exit 1
fi

echo ""
echo "📦 To install the package:"
echo "  pip install -e ."
echo ""
echo "🧪 To run tests:"
echo "  python comprehensive_test.py"