#!/usr/bin/env python3
"""
ControlNet Example for OllamaDiffuser

This example demonstrates how to use ControlNet models for controlled image generation.
ControlNet allows you to guide the image generation process using control images like
edge maps, depth maps, pose keypoints, etc.

Usage:
    python examples/controlnet_example.py
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main example function"""
    print("🎨 ControlNet Example for OllamaDiffuser")
    print("=" * 50)
    
    # Step 1: List available ControlNet models
    print("\n📋 Available ControlNet models:")
    available_models = model_manager.list_available_models()
    controlnet_models = [model for model in available_models if "controlnet" in model]
    
    for model in controlnet_models:
        info = model_manager.get_model_info(model)
        if info:
            print(f"  • {model} ({info.get('controlnet_type', 'unknown')} type)")
            print(f"    Base model: {info.get('base_model', 'unknown')}")
            print(f"    Installed: {'✅' if info.get('installed', False) else '❌'}")
    
    if not controlnet_models:
        print("  No ControlNet models found in registry")
        return
    
    # Step 2: Check if we have a ControlNet model installed
    installed_controlnet = [model for model in controlnet_models 
                           if model_manager.is_model_installed(model)]
    
    if not installed_controlnet:
        print("\n⚠️  No ControlNet models installed.")
        print("To install a ControlNet model, run:")
        print("  ollamadiffuser pull controlnet-canny-sd15")
        print("\nNote: You'll also need the base model (stable-diffusion-1.5)")
        return
    
    # Use the first available ControlNet model
    model_name = installed_controlnet[0]
    print(f"\n🚀 Using ControlNet model: {model_name}")
    
    # Step 3: Load the ControlNet model
    print("📦 Loading ControlNet model...")
    if not model_manager.load_model(model_name):
        print("❌ Failed to load ControlNet model")
        return
    
    print("✅ ControlNet model loaded successfully!")
    
    # Step 4: Demonstrate ControlNet preprocessors
    print("\n🔧 Available ControlNet preprocessors:")
    available_types = controlnet_preprocessor.get_available_types()
    for control_type in available_types:
        print(f"  • {control_type}")
    
    # Step 5: Create a sample control image (if you have an input image)
    # For this example, we'll create a simple test image
    print("\n🖼️  Creating sample control image...")
    
    # Create a simple test image with some geometric shapes
    test_image = Image.new('RGB', (512, 512), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    
    # Draw some shapes for edge detection
    draw.rectangle([100, 100, 200, 200], outline='black', width=3)
    draw.circle([300, 150], 50, outline='black', width=3)
    draw.line([50, 300, 450, 350], fill='black', width=3)
    
    # Save test image
    test_image_path = "test_control_image.png"
    test_image.save(test_image_path)
    print(f"📁 Saved test image: {test_image_path}")
    
    # Step 6: Preprocess the control image
    model_info = model_manager.get_model_info(model_name)
    control_type = model_info.get('controlnet_type', 'canny')
    
    print(f"\n⚙️  Preprocessing image for {control_type} ControlNet...")
    try:
        processed_image = controlnet_preprocessor.preprocess(test_image, control_type)
        processed_path = f"processed_{control_type}_image.png"
        processed_image.save(processed_path)
        print(f"📁 Saved processed image: {processed_path}")
    except Exception as e:
        print(f"❌ Failed to preprocess image: {e}")
        return
    
    # Step 7: Generate image with ControlNet
    print("\n🎨 Generating image with ControlNet...")
    
    try:
        # Get the inference engine
        engine = model_manager.loaded_model
        
        # Generate image with ControlNet
        generated_image = engine.generate_image(
            prompt="a beautiful landscape with mountains and trees, highly detailed, photorealistic",
            negative_prompt="low quality, blurry, distorted",
            num_inference_steps=20,
            guidance_scale=7.0,
            width=512,
            height=512,
            control_image=test_image,  # Use original image for preprocessing
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0
        )
        
        # Save generated image
        output_path = f"controlnet_generated_{control_type}.png"
        generated_image.save(output_path)
        print(f"✅ Generated image saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to generate image: {e}")
        return
    
    # Step 8: Cleanup and summary
    print("\n🧹 Cleaning up...")
    model_manager.unload_model()
    
    print("\n🎉 ControlNet example completed successfully!")
    print("\nGenerated files:")
    print(f"  • {test_image_path} - Original test image")
    print(f"  • {processed_path} - Preprocessed control image")
    print(f"  • {output_path} - Final generated image")
    
    print("\n💡 Tips:")
    print("  • Try different ControlNet types (canny, depth, openpose, etc.)")
    print("  • Adjust controlnet_conditioning_scale to control influence strength")
    print("  • Use control_guidance_start/end to apply control only during certain steps")
    print("  • Experiment with different prompts and base images")

if __name__ == "__main__":
    main() 