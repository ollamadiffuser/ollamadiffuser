#!/usr/bin/env python3
"""
OllamaDiffuser Quick Start Script
Automatically installs dependencies, tests environment, and provides quick start guidance
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run command and show progress"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ {description} completed")
            return True
        else:
            print(f"  ❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ {description} exception: {e}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} version is suitable")
        return True
    else:
        print(f"  ❌ Python version too low: {version.major}.{version.minor}.{version.micro}")
        print("     Python 3.8 or higher required")
        return False

def install_dependencies():
    """Install dependencies"""
    print("\n📦 Installing dependency packages...")
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("  ⚠️  Recommend running in virtual environment")
        print("     Create virtual environment: python -m venv venv")
        print("     Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        
        response = input("  Continue installing in current environment? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements.txt
    if Path("requirements.txt").exists():
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependency packages"):
            return False
    else:
        print("  ❌ requirements.txt file does not exist")
        return False
    
    # Install project itself
    if not run_command(f"{sys.executable} -m pip install -e .", "Installing OllamaDiffuser"):
        return False
    
    return True

def test_installation():
    """Test installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test import
        import ollamadiffuser
        print("  ✅ OllamaDiffuser import successful")
        
        # Run installation test
        if Path("test_installation.py").exists():
            result = subprocess.run([sys.executable, "test_installation.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("  ✅ Installation test passed")
                return True
            else:
                print("  ❌ Installation test failed")
                print(result.stdout)
                return False
        else:
            print("  ✅ Basic import test passed")
            return True
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print("\n⚙️  Environment setup...")
    
    # Check HuggingFace token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("  ⚠️  HuggingFace token not set")
        print("     Some models require HuggingFace account to download")
        print("     Setup method: export HF_TOKEN=your_token_here")
        
        token = input("  Please enter your HuggingFace token (optional): ").strip()
        if token:
            os.environ['HF_TOKEN'] = token
            print("  ✅ HuggingFace token set (temporary)")
        else:
            print("  ⚠️  Skipping HuggingFace token setup")
    else:
        print("  ✅ HuggingFace token already set")

def show_quick_start_guide():
    """Show quick start guide"""
    print("\n🚀 Quick Start Guide")
    print("=" * 50)
    
    print("\n1. View available models:")
    print("   ollamadiffuser list")
    
    print("\n2. Download model (recommended to start with smaller models):")
    print("   ollamadiffuser pull stable-diffusion-1.5")
    print("   # Or larger models:")
    print("   ollamadiffuser pull stable-diffusion-3.5-medium")
    
    print("\n3. Run model service:")
    print("   ollamadiffuser run stable-diffusion-1.5")
    
    print("\n4. Generate image in another terminal:")
    print('   curl -X POST http://localhost:8000/api/generate \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"prompt": "A beautiful sunset over mountains"}\' \\')
    print('     --output image.png')
    
    print("\n5. Or start Web UI:")
    print("   python -m ollamadiffuser --mode ui")
    print("   Then visit: http://localhost:8001")
    
    print("\n6. Other useful commands:")
    print("   ollamadiffuser show MODEL_NAME     # View model information")
    print("   ollamadiffuser ps                  # View running status")
    print("   ollamadiffuser unload              # Unload model")
    print("   ollamadiffuser rm MODEL_NAME       # Remove model")
    
    print("\n📚 More information:")
    print("   - Project documentation: README.md")
    print("   - Project structure: PROJECT_STRUCTURE.md")
    print("   - Demo script: python demo.py")

def main():
    """Main function"""
    print("🎨 OllamaDiffuser Quick Start")
    print("=" * 50)
    print("An Ollama-like image generation model management tool")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed, please check error information")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed, please check error information")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Show quick start guide
    show_quick_start_guide()
    
    print("\n🎉 Installation completed!")
    print("You can now start using OllamaDiffuser!")

if __name__ == "__main__":
    main() 