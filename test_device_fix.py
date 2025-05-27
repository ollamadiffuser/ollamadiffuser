#!/usr/bin/env python3
"""
Test script to verify device detection and FLUX model loading fixes
"""

import torch
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollamadiffuser.core.inference.engine import InferenceEngine
from ollamadiffuser.core.config.settings import ModelConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_device_detection():
    """Test device detection"""
    print("=== Testing Device Detection ===")
    
    engine = InferenceEngine()
    device = engine._get_device()
    
    print(f"Detected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    return device

def test_generator_creation(device):
    """Test generator creation for different devices"""
    print(f"\n=== Testing Generator Creation for {device} ===")
    
    try:
        if device == "cpu":
            generator = torch.Generator().manual_seed(42)
            print("✅ CPU generator created successfully")
        else:
            generator = torch.Generator(device=device).manual_seed(42)
            print(f"✅ {device.upper()} generator created successfully")
        
        # Test generator device
        if hasattr(generator, 'device'):
            print(f"Generator device: {generator.device}")
        else:
            print("Generator device: CPU (no device attribute)")
            
        return True
    except Exception as e:
        print(f"❌ Generator creation failed: {e}")
        return False

def test_flux_model_config():
    """Test FLUX model configuration"""
    print("\n=== Testing FLUX Model Configuration ===")
    
    try:
        # Create a mock model config for FLUX
        model_config = ModelConfig(
            name="flux.1-dev",
            model_type="flux",
            path="/Users/xiasun/.ollamadiffuser/models/flux.1-dev",
            variant="bf16",
            parameters={
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "max_sequence_length": 512
            }
        )
        
        print(f"Model config created: {model_config.name}")
        print(f"Model type: {model_config.model_type}")
        print(f"Model path: {model_config.path}")
        print(f"Variant: {model_config.variant}")
        
        return model_config
    except Exception as e:
        print(f"❌ Model config creation failed: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Testing FLUX Device Fixes\n")
    
    # Test 1: Device detection
    device = test_device_detection()
    
    # Test 2: Generator creation
    generator_ok = test_generator_creation(device)
    
    # Test 3: Model configuration
    model_config = test_flux_model_config()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Device detected: {device}")
    print(f"Generator creation: {'✅ PASS' if generator_ok else '❌ FAIL'}")
    print(f"Model config: {'✅ PASS' if model_config else '❌ FAIL'}")
    
    if device == "cpu":
        print("\n⚠️  Note: CPU device detected. FLUX inference will be slow.")
        print("💡 Consider using a GPU with at least 12GB VRAM for better performance.")
    
    return device, generator_ok, model_config is not None

if __name__ == "__main__":
    main() 