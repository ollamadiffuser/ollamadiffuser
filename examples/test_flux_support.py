#!/usr/bin/env python3
"""
Test script for FLUX.1-dev model support in OllamaDiffuser
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.config.settings import settings

def test_flux_model_registry():
    """Test if FLUX.1-dev is in the model registry"""
    print("🔍 Testing FLUX.1-dev model registry...")
    
    available_models = model_manager.list_available_models()
    print(f"Available models: {available_models}")
    
    if "flux.1-dev" in available_models:
        print("✅ FLUX.1-dev found in model registry")
        
        # Get model info
        model_info = model_manager.get_model_info("flux.1-dev")
        print(f"📋 Model info: {model_info}")
        
        # Check hardware requirements
        hw_req = model_info.get("hardware_requirements", {})
        print(f"💻 Hardware requirements:")
        print(f"   - Min VRAM: {hw_req.get('min_vram_gb', 'N/A')}GB")
        print(f"   - Recommended VRAM: {hw_req.get('recommended_vram_gb', 'N/A')}GB")
        print(f"   - Min RAM: {hw_req.get('min_ram_gb', 'N/A')}GB")
        print(f"   - Disk space: {hw_req.get('disk_space_gb', 'N/A')}GB")
        
        # Check license info
        license_info = model_info.get("license_info", {})
        if license_info:
            print(f"📄 License: {license_info.get('type', 'N/A')}")
            print(f"   - Requires agreement: {license_info.get('requires_agreement', False)}")
            print(f"   - Commercial use: {license_info.get('commercial_use', 'Unknown')}")
        
        return True
    else:
        print("❌ FLUX.1-dev not found in model registry")
        return False

def test_flux_pipeline_support():
    """Test if FluxPipeline is supported in inference engine"""
    print("\n🔍 Testing FluxPipeline support...")
    
    try:
        from ollamadiffuser.core.inference.engine import InferenceEngine
        from diffusers import FluxPipeline
        
        engine = InferenceEngine()
        pipeline_class = engine._get_pipeline_class("flux")
        
        if pipeline_class == FluxPipeline:
            print("✅ FluxPipeline correctly mapped for 'flux' model type")
            return True
        else:
            print(f"❌ FluxPipeline not mapped correctly. Got: {pipeline_class}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing pipeline support: {e}")
        return False

def test_hf_token_setup():
    """Test HuggingFace token setup"""
    print("\n🔍 Testing HuggingFace token setup...")
    
    hf_token = os.environ.get('HF_TOKEN') or settings.hf_token
    
    if hf_token:
        print("✅ HuggingFace token is configured")
        print(f"   Token (first 10 chars): {hf_token[:10]}...")
        return True
    else:
        print("⚠️  HuggingFace token not found")
        print("   To use FLUX.1-dev, you need to:")
        print("   1. Create an account at https://huggingface.co")
        print("   2. Accept the FLUX.1-dev license agreement")
        print("   3. Set HF_TOKEN environment variable or configure in settings")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n🔍 Testing dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Check for CUDA/MPS
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("⚠️  Only CPU available (FLUX.1-dev will be very slow)")
        
        from diffusers import FluxPipeline
        print("✅ Diffusers with FluxPipeline support")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FLUX.1-dev support in OllamaDiffuser\n")
    
    tests = [
        test_flux_model_registry,
        test_flux_pipeline_support,
        test_dependencies,
        test_hf_token_setup,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! FLUX.1-dev support is ready.")
        print("\n📝 To use FLUX.1-dev:")
        print("   1. Ensure you have a HuggingFace token with FLUX.1-dev access")
        print("   2. Run: ollamadiffuser pull flux.1-dev")
        print("   3. Run: ollamadiffuser run flux.1-dev")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 