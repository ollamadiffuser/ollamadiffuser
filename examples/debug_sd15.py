#!/usr/bin/env python3
"""
Debug script for Stable Diffusion 1.5 black image issue
"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sd15_generation():
    """Test SD 1.5 generation with debugging"""
    try:
        from ollamadiffuser.core.models.manager import model_manager
        from ollamadiffuser.core.config.settings import settings
        
        logger.info("=== SD 1.5 Debug Test ===")
        
        # Check if SD 1.5 is installed
        if not model_manager.is_model_installed("stable-diffusion-1.5"):
            logger.error("Stable Diffusion 1.5 is not installed")
            logger.info("Please install it first: ollamadiffuser pull stable-diffusion-1.5")
            return False
        
        # Unload any current model first
        if model_manager.is_model_loaded():
            logger.info("Unloading current model...")
            model_manager.unload_model()
        
        # Load SD 1.5
        logger.info("Loading Stable Diffusion 1.5...")
        if not model_manager.load_model("stable-diffusion-1.5"):
            logger.error("Failed to load Stable Diffusion 1.5")
            return False
        
        # Get the inference engine
        engine = model_manager.loaded_model
        
        # Check safety checker status
        logger.info("=== Safety Checker Status ===")
        logger.info(f"Pipeline has safety_checker: {hasattr(engine.pipeline, 'safety_checker')}")
        if hasattr(engine.pipeline, 'safety_checker'):
            logger.info(f"Safety checker value: {engine.pipeline.safety_checker}")
        
        logger.info(f"Pipeline has feature_extractor: {hasattr(engine.pipeline, 'feature_extractor')}")
        if hasattr(engine.pipeline, 'feature_extractor'):
            logger.info(f"Feature extractor value: {engine.pipeline.feature_extractor}")
        
        logger.info(f"Pipeline has requires_safety_checker: {hasattr(engine.pipeline, 'requires_safety_checker')}")
        if hasattr(engine.pipeline, 'requires_safety_checker'):
            logger.info(f"Requires safety checker: {engine.pipeline.requires_safety_checker}")
        
        # Test generation with simple prompt
        logger.info("=== Testing Image Generation ===")
        test_prompts = [
            "a red apple",
            "blue sky",
            "green grass",
            "simple landscape"
        ]
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Testing prompt {i+1}: '{prompt}'")
            try:
                image = engine.generate_image(
                    prompt=prompt,
                    negative_prompt="",
                    num_inference_steps=10,  # Use fewer steps for faster testing
                    guidance_scale=7.0,
                    width=512,
                    height=512
                )
                
                # Save test image
                output_path = f"debug_test_{i+1}.png"
                image.save(output_path)
                logger.info(f"Image saved to: {output_path}")
                
                # Check if image is black
                import numpy as np
                img_array = np.array(image)
                if np.all(img_array == 0):
                    logger.error(f"Image {i+1} is completely black!")
                else:
                    mean_val = np.mean(img_array)
                    logger.info(f"Image {i+1} looks good - mean pixel value: {mean_val:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Debug test failed: {e}")
        return False

def check_diffusers_version():
    """Check diffusers version for compatibility"""
    try:
        import diffusers
        logger.info(f"Diffusers version: {diffusers.__version__}")
        
        # Check if this version has known safety checker issues
        version = diffusers.__version__
        if version.startswith("0.21") or version.startswith("0.22"):
            logger.warning("This diffusers version may have safety checker issues")
            logger.info("Consider upgrading: pip install --upgrade diffusers")
        
    except ImportError:
        logger.error("Diffusers not installed")

def main():
    """Main debug function"""
    logger.info("Starting SD 1.5 debug session...")
    
    # Check diffusers version
    check_diffusers_version()
    
    # Run debug test
    if test_sd15_generation():
        logger.info("Debug test completed successfully")
    else:
        logger.error("Debug test failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 