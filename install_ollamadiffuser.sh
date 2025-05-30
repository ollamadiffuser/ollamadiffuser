#!/bin/bash

echo "🎨 OllamaDiffuser Installation Script"
echo "===================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
MIN_VERSION="3.10"

if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION detected. OllamaDiffuser requires Python 3.10+."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check pip
if ! command_exists pip3; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 detected"

# Install OllamaDiffuser with all dependencies
echo ""
echo "📦 Installing OllamaDiffuser..."
pip3 install --upgrade pip

# Try to install with all optional dependencies
echo "📦 Installing OllamaDiffuser with full dependencies..."
if pip3 install ollamadiffuser[full]; then
    echo "✅ OllamaDiffuser installed successfully!"
else
    echo "⚠️  Full installation failed, trying basic installation..."
    if pip3 install ollamadiffuser; then
        echo "✅ Basic OllamaDiffuser installed"
        echo "📦 Installing missing dependencies manually..."
        pip3 install "opencv-python>=4.8.0" || echo "⚠️ OpenCV installation failed"
        pip3 install "controlnet-aux>=0.0.7" || echo "⚠️ ControlNet-aux installation failed"
    else
        echo "❌ Failed to install OllamaDiffuser"
        exit 1
    fi
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
if command_exists ollamadiffuser; then
    echo "✅ OllamaDiffuser command is available"
    
    # Run dependency verification
    echo ""
    echo "🩺 Running system diagnostics..."
    ollamadiffuser doctor
    
    echo ""
    echo "🎉 Installation complete!"
    echo ""
    echo "Quick start commands:"
    echo "  ollamadiffuser pull flux.1-schnell    # Download a model"
    echo "  ollamadiffuser run flux.1-schnell     # Start the model"
    echo "  ollamadiffuser --mode ui              # Start web interface"
    echo ""
    echo "For help: ollamadiffuser --help"
else
    echo "❌ OllamaDiffuser command not found after installation"
    exit 1
fi 