#!/bin/bash

# OllamaDiffuser PyPI Publishing Script

echo "🚀 Publishing OllamaDiffuser to PyPI"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build the package
echo "📦 Building package..."
python -m build

# Check the package
echo "🔍 Checking package..."
python -m twine check dist/*

echo "📋 Package built successfully!"
echo ""
echo "Next steps:"
echo "1. Test upload to TestPyPI (optional):"
echo "   python -m twine upload --repository testpypi dist/*"
echo ""
echo "2. Upload to PyPI:"
echo "   python -m twine upload dist/*"
echo ""
echo "Note: You'll need PyPI account credentials or API token" 