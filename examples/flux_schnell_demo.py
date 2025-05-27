#!/usr/bin/env python3
"""
FLUX.1-schnell Demo - Fast High-Quality Image Generation
"""

import sys
import os
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.models.manager import model_manager

def demo_flux_schnell():
    """Demonstrate FLUX.1-schnell usage"""
    print("🚀 FLUX.1-schnell Demo - Fast High-Quality Image Generation\n")
    
    # Check if FLUX.1-schnell is available
    if "flux.1-schnell" not in model_manager.list_available_models():
        print("❌ FLUX.1-schnell not found in model registry")
        return False
    
    # Get model info
    model_info = model_manager.get_model_info("flux.1-schnell")
    print("📋 FLUX.1-schnell Information:")
    print(f"   Repository: {model_info['repo_id']}")
    print(f"   License: {model_info['license_info']['type']}")
    print(f"   Commercial use: {'✅ Allowed' if model_info['license_info']['commercial_use'] else '❌ Not allowed'}")
    print(f"   HuggingFace token required: {'❌ No' if not model_info['license_info']['requires_agreement'] else '✅ Yes'}")
    
    # Show optimal parameters
    params = model_info['parameters']
    print(f"\n⚡ Optimal Parameters:")
    print(f"   Inference steps: {params['num_inference_steps']} (vs 50 for FLUX.1-dev)")
    print(f"   Guidance scale: {params['guidance_scale']} (distilled model)")
    print(f"   Max sequence length: {params['max_sequence_length']}")
    
    print(f"\n🎯 Key Benefits:")
    print(f"   ✅ 12x faster than FLUX.1-dev (4 steps vs 50)")
    print(f"   ✅ Same image quality")
    print(f"   ✅ No HuggingFace token required")
    print(f"   ✅ Commercial use allowed")
    print(f"   ✅ Apache 2.0 license")
    
    # Check if installed
    if model_manager.is_model_installed("flux.1-schnell"):
        print(f"\n✅ FLUX.1-schnell is installed and ready to use!")
        
        # Show example usage
        print(f"\n📝 Example Usage:")
        print(f"   # Start the model server")
        print(f"   ollamadiffuser run flux.1-schnell")
        print(f"   ")
        print(f"   # Generate image via API")
        print(f"   curl -X POST http://localhost:8000/api/generate \\")
        print(f"     -H 'Content-Type: application/json' \\")
        print(f"     -d '{{")
        print(f"       \"prompt\": \"A beautiful sunset over mountains\",")
        print(f"       \"num_inference_steps\": 4,")
        print(f"       \"guidance_scale\": 0.0,")
        print(f"       \"width\": 1024,")
        print(f"       \"height\": 1024")
        print(f"     }}' \\")
        print(f"     --output image.png")
        
        # Test generation if model is loaded
        if model_manager.is_model_loaded() and model_manager.get_current_model() == "flux.1-schnell":
            print(f"\n🎨 Generating test image...")
            try:
                start_time = time.time()
                image = model_manager.loaded_model.generate_image(
                    "A cute robot in a garden",
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    width=1024,
                    height=1024
                )
                end_time = time.time()
                
                # Save image
                output_path = "flux_schnell_demo.png"
                image.save(output_path)
                
                print(f"✅ Image generated in {end_time - start_time:.2f} seconds!")
                print(f"💾 Saved to: {output_path}")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        else:
            print(f"\n💡 To test generation, first load the model:")
            print(f"   ollamadiffuser load flux.1-schnell")
    else:
        print(f"\n⬇️ FLUX.1-schnell is not installed yet.")
        print(f"\n📥 To install:")
        print(f"   ollamadiffuser pull flux.1-schnell")
        print(f"   ")
        print(f"   # No HuggingFace token required!")
        print(f"   # Download size: ~15GB")
    
    return True

def compare_flux_models():
    """Compare FLUX.1-schnell vs FLUX.1-dev"""
    print(f"\n📊 FLUX Model Comparison:")
    print(f"{'Aspect':<25} {'FLUX.1-schnell':<20} {'FLUX.1-dev':<20}")
    print("-" * 70)
    
    schnell_info = model_manager.get_model_info("flux.1-schnell")
    dev_info = model_manager.get_model_info("flux.1-dev")
    
    if schnell_info and dev_info:
        schnell_params = schnell_info['parameters']
        dev_params = dev_info['parameters']
        schnell_license = schnell_info['license_info']
        dev_license = dev_info['license_info']
        
        print(f"{'Inference Steps':<25} {schnell_params['num_inference_steps']:<20} {dev_params['num_inference_steps']:<20}")
        print(f"{'Guidance Scale':<25} {schnell_params['guidance_scale']:<20} {dev_params['guidance_scale']:<20}")
        print(f"{'Generation Speed':<25} {'⚡ Very Fast':<20} {'🐌 Slow':<20}")
        print(f"{'License':<25} {schnell_license['type']:<20} {dev_license['type'][:20]:<20}")
        print(f"{'Commercial Use':<25} {'✅ Yes':<20} {'❌ No':<20}")
        print(f"{'HF Token Required':<25} {'❌ No':<20} {'✅ Yes':<20}")
        print(f"{'Image Quality':<25} {'🎯 Excellent':<20} {'🎯 Excellent':<20}")
        
        print(f"\n🏆 Winner for most use cases: FLUX.1-schnell")
        print(f"   - Same quality, 12x faster")
        print(f"   - No licensing restrictions")
        print(f"   - Easier to set up")

def main():
    """Main demo function"""
    try:
        demo_flux_schnell()
        compare_flux_models()
        
        print(f"\n🎉 FLUX.1-schnell Demo Complete!")
        print(f"\n🚀 Ready to generate high-quality images in seconds!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 