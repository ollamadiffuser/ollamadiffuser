#!/usr/bin/env python3
"""
OllamaDiffuser Installation Test Script
Verify that all components are correctly installed and configured
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_imports():
    """Test all required package imports"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'torch',
        'diffusers',
        'transformers',
        'accelerate',
        'fastapi',
        'uvicorn',
        'huggingface_hub',
        'PIL',
        'click',
        'rich',
        'pydantic'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'huggingface_hub':
                importlib.import_module('huggingface_hub')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages imported successfully!")
        return True

def test_ollamadiffuser_imports():
    """Test OllamaDiffuser component imports"""
    print("\n🔍 Testing OllamaDiffuser component imports...")
    
    components = [
        'ollamadiffuser.core.config.settings',
        'ollamadiffuser.core.models.manager',
        'ollamadiffuser.core.inference.engine',
        'ollamadiffuser.api.server',
        'ollamadiffuser.cli.main',
        'ollamadiffuser.ui.web'
    ]
    
    failed_imports = []
    
    for component in components:
        try:
            importlib.import_module(component)
            print(f"  ✅ {component}")
        except ImportError as e:
            print(f"  ❌ {component}: {e}")
            failed_imports.append(component)
    
    if failed_imports:
        print(f"\n❌ Failed component imports: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All OllamaDiffuser components imported successfully!")
        return True

def test_hardware():
    """Test hardware support"""
    print("\n🔍 Testing hardware support...")
    
    try:
        import torch
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  CUDA not available")
        
        # Test MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon) available")
        else:
            print("  ⚠️  MPS not available")
        
        print(f"  ✅ CPU available: {torch.get_num_threads()} threads")
        print(f"  ✅ PyTorch version: {torch.__version__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hardware test failed: {e}")
        return False

def test_cli():
    """Test CLI commands"""
    print("\n🔍 Testing CLI commands...")
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, '-m', 'ollamadiffuser.cli.main', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ CLI help command works properly")
        else:
            print(f"  ❌ CLI help command failed: {result.stderr}")
            return False
        
        # Test list command
        result = subprocess.run([
            sys.executable, '-m', 'ollamadiffuser.cli.main', 'list'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ CLI list command works properly")
        else:
            print(f"  ❌ CLI list command failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False

def test_config():
    """Test configuration system"""
    print("\n🔍 Testing configuration system...")
    
    try:
        from ollamadiffuser.core.config.settings import settings
        
        print(f"  ✅ Configuration directory: {settings.config_dir}")
        print(f"  ✅ Models directory: {settings.models_dir}")
        print(f"  ✅ Cache directory: {settings.cache_dir}")
        print(f"  ✅ Server configuration: {settings.server.host}:{settings.server.port}")
        
        # Check if directory is created
        if settings.config_dir.exists():
            print("  ✅ Configuration directory created")
        else:
            print("  ❌ Configuration directory not created")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_model_manager():
    """Test model manager"""
    print("\n🔍 Testing model manager...")
    
    try:
        from ollamadiffuser.core.models.manager import model_manager
        
        # Test available models list
        available_models = model_manager.list_available_models()
        print(f"  ✅ Available models: {len(available_models)} models")
        for model in available_models:
            print(f"    • {model}")
        
        # Test installed models list
        installed_models = model_manager.list_installed_models()
        print(f"  ✅ Installed models: {len(installed_models)} models")
        
        # Test model information
        if available_models:
            model_name = available_models[0]
            info = model_manager.get_model_info(model_name)
            if info:
                print(f"  ✅ Model information retrieved successfully: {model_name}")
            else:
                print(f"  ❌ Failed to retrieve model information: {model_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model manager test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 OllamaDiffuser Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("Component imports", test_ollamadiffuser_imports),
        ("Hardware support", test_hardware),
        ("Configuration system", test_config),
        ("Model manager", test_model_manager),
        ("CLI commands", test_cli),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test exception: {e}")
    
    print(f"\n📊 Test results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! OllamaDiffuser installation successful!")
        print("\n🚀 Quick start:")
        print("1. View available models: ollamadiffuser list")
        print("2. Download model: ollamadiffuser pull stable-diffusion-3.5-medium")
        print("3. Run service: ollamadiffuser run stable-diffusion-3.5-medium")
        return True
    else:
        print("❌ Some tests failed, please check installation!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 