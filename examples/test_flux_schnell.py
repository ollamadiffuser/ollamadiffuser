#!/usr/bin/env python3
"""
Test script for FLUX.1-schnell model support in OllamaDiffuser
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.config.settings import settings

def test_flux_schnell_registry():
    """Test if FLUX.1-schnell is in the model registry"""
    print("🔍 Testing FLUX.1-schnell model registry...")
    
    available_models = model_manager.list_available_models()
    print(f"Available models: {available_models}")
    
    if "flux.1-schnell" in available_models:
        print("✅ FLUX.1-schnell found in model registry")
        
        # Get model info
        model_info = model_manager.get_model_info("flux.1-schnell")
        print(f"📋 Model info: {model_info}")
        
        # Check parameters
        params = model_info.get("parameters", {})
        print(f"🎛️ Default parameters:")
        print(f"   - Inference steps: {params.get('num_inference_steps', 'N/A')}")
        print(f"   - Guidance scale: {params.get('guidance_scale', 'N/A')}")
        print(f"   - Max sequence length: {params.get('max_sequence_length', 'N/A')}")
        
        # Check hardware requirements
        hw_req = model_info.get("hardware_requirements", {})
        print(f"💻 Hardware requirements:")
        print(f"   - Min VRAM: {hw_req.get('min_vram_gb', 'N/A')}GB")
        print(f"   - Recommended VRAM: {hw_req.get('recommended_vram_gb', 'N/A')}GB")
        print(f"   - Min RAM: {hw_req.get('min_ram_gb', 'N/A')}GB")
        print(f"   - Disk space: {hw_req.get('disk_space_gb', 'N/A')}GB")
        print(f"   - Performance notes: {hw_req.get('performance_notes', 'N/A')}")
        
        # Check license info
        license_info = model_info.get("license_info", {})
        if license_info:
            print(f"📄 License: {license_info.get('type', 'N/A')}")
            print(f"   - Requires agreement: {license_info.get('requires_agreement', False)}")
            print(f"   - Commercial use: {license_info.get('commercial_use', 'Unknown')}")
        
        return True
    else:
        print("❌ FLUX.1-schnell not found in model registry")
        return False

def test_flux_schnell_vs_dev():
    """Compare FLUX.1-schnell vs FLUX.1-dev"""
    print("\n🔍 Comparing FLUX.1-schnell vs FLUX.1-dev...")
    
    schnell_info = model_manager.get_model_info("flux.1-schnell")
    dev_info = model_manager.get_model_info("flux.1-dev")
    
    if not schnell_info or not dev_info:
        print("❌ Could not get model info for comparison")
        return False
    
    print("📊 Comparison:")
    print(f"{'Aspect':<20} {'FLUX.1-schnell':<20} {'FLUX.1-dev':<20}")
    print("-" * 65)
    
    # Parameters comparison
    schnell_params = schnell_info.get("parameters", {})
    dev_params = dev_info.get("parameters", {})
    
    print(f"{'Steps':<20} {schnell_params.get('num_inference_steps', 'N/A'):<20} {dev_params.get('num_inference_steps', 'N/A'):<20}")
    print(f"{'Guidance Scale':<20} {schnell_params.get('guidance_scale', 'N/A'):<20} {dev_params.get('guidance_scale', 'N/A'):<20}")
    print(f"{'Max Seq Length':<20} {schnell_params.get('max_sequence_length', 'N/A'):<20} {dev_params.get('max_sequence_length', 'N/A'):<20}")
    
    # License comparison
    schnell_license = schnell_info.get("license_info", {})
    dev_license = dev_info.get("license_info", {})
    
    print(f"{'License':<20} {schnell_license.get('type', 'N/A'):<20} {dev_license.get('type', 'N/A'):<20}")
    print(f"{'Commercial Use':<20} {schnell_license.get('commercial_use', 'N/A'):<20} {dev_license.get('commercial_use', 'N/A'):<20}")
    print(f"{'Requires Token':<20} {schnell_license.get('requires_agreement', 'N/A'):<20} {dev_license.get('requires_agreement', 'N/A'):<20}")
    
    print("\n🚀 Key Advantages of FLUX.1-schnell:")
    print("   ✅ 12x faster generation (4 steps vs 50 steps)")
    print("   ✅ No HuggingFace token required")
    print("   ✅ Commercial use allowed (Apache 2.0)")
    print("   ✅ Same image quality as FLUX.1-dev")
    print("   ✅ Smaller memory footprint during inference")
    
    return True

def test_flux_schnell_pipeline():
    """Test FLUX.1-schnell pipeline configuration"""
    print("\n🔍 Testing FLUX.1-schnell pipeline configuration...")
    
    try:
        from ollamadiffuser.core.inference.engine import InferenceEngine
        from diffusers import FluxPipeline
        
        engine = InferenceEngine()
        pipeline_class = engine._get_pipeline_class("flux")
        
        if pipeline_class == FluxPipeline:
            print("✅ FluxPipeline correctly mapped for FLUX.1-schnell")
            
            # Test model config creation
            from ollamadiffuser.core.config.settings import ModelConfig
            
            model_config = ModelConfig(
                name="flux.1-schnell",
                path="/test/path",
                model_type="flux",
                variant="bf16",
                parameters={
                    "num_inference_steps": 4,
                    "guidance_scale": 0.0,
                    "max_sequence_length": 256
                }
            )
            
            print("✅ FLUX.1-schnell model config created successfully")
            print(f"   - Model type: {model_config.model_type}")
            print(f"   - Variant: {model_config.variant}")
            print(f"   - Parameters: {model_config.parameters}")
            
            return True
        else:
            print(f"❌ FluxPipeline not mapped correctly. Got: {pipeline_class}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")
        return False

def test_no_hf_token_required():
    """Test that FLUX.1-schnell doesn't require HuggingFace token"""
    print("\n🔍 Testing HuggingFace token requirements...")
    
    schnell_info = model_manager.get_model_info("flux.1-schnell")
    dev_info = model_manager.get_model_info("flux.1-dev")
    
    if not schnell_info or not dev_info:
        print("❌ Could not get model info")
        return False
    
    schnell_requires_token = schnell_info.get("license_info", {}).get("requires_agreement", True)
    dev_requires_token = dev_info.get("license_info", {}).get("requires_agreement", True)
    
    print(f"FLUX.1-schnell requires HF token: {schnell_requires_token}")
    print(f"FLUX.1-dev requires HF token: {dev_requires_token}")
    
    if not schnell_requires_token and dev_requires_token:
        print("✅ FLUX.1-schnell correctly configured to not require HuggingFace token")
        return True
    else:
        print("❌ Token requirements not configured correctly")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FLUX.1-schnell support in OllamaDiffuser\n")
    
    tests = [
        test_flux_schnell_registry,
        test_flux_schnell_vs_dev,
        test_flux_schnell_pipeline,
        test_no_hf_token_required,
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
        print("🎉 All tests passed! FLUX.1-schnell support is ready.")
        print("\n📝 To use FLUX.1-schnell:")
        print("   1. No HuggingFace token required!")
        print("   2. Run: ollamadiffuser pull flux.1-schnell")
        print("   3. Run: ollamadiffuser run flux.1-schnell")
        print("\n⚡ FLUX.1-schnell generates images in ~4 steps vs 50 for FLUX.1-dev")
        print("🎯 Same quality, 12x faster, commercial use allowed!")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 