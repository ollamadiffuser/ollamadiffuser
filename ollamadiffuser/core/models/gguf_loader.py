"""
GGUF Model Loader and Interface

This module provides support for loading and running GGUF quantized models,
specifically for FLUX.1-dev-gguf variants using stable-diffusion.cpp Python bindings.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

try:
    from stable_diffusion_cpp import StableDiffusion
    GGUF_AVAILABLE = True
    logger.info("stable-diffusion-cpp-python is available")
except ImportError:
    StableDiffusion = None
    GGUF_AVAILABLE = False
    logger.warning("stable-diffusion-cpp-python not available. GGUF models will not work.")

class GGUFModelLoader:
    """Loader for GGUF quantized diffusion models using stable-diffusion.cpp"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_config = None
        self.loaded_model_name = None
        self.stable_diffusion = None
        
    def is_gguf_model(self, model_name: str, model_config: Dict[str, Any]) -> bool:
        """Check if a model is a GGUF model"""
        variant = model_config.get('variant', '')
        return 'gguf' in variant.lower() or model_name.endswith('-gguf') or 'gguf' in model_name.lower()
    
    def get_gguf_file_path(self, model_dir: Path, variant: str) -> Optional[Path]:
        """Find the appropriate GGUF file based on variant"""
        if not model_dir.exists():
            return None
            
        # Map variant to actual file names
        variant_mapping = {
            # FLUX.1-dev variants
            'gguf-q2k': 'flux1-dev-Q2_K.gguf',
            'gguf-q3ks': 'flux1-dev-Q3_K_S.gguf', 
            'gguf-q4ks': 'flux1-dev-Q4_K_S.gguf',
            'gguf-q4-0': 'flux1-dev-Q4_0.gguf',
            'gguf-q4-1': 'flux1-dev-Q4_1.gguf',
            'gguf-q5ks': 'flux1-dev-Q5_K_S.gguf',
            'gguf-q5-0': 'flux1-dev-Q5_0.gguf',
            'gguf-q5-1': 'flux1-dev-Q5_1.gguf',
            'gguf-q6k': 'flux1-dev-Q6_K.gguf',
            'gguf-q8': 'flux1-dev-Q8_0.gguf',
            'gguf-f16': 'flux1-dev-F16.gguf',
            
            # FLUX.1-schnell variants
            'gguf-schnell': 'flux1-schnell-F16.gguf',  # Default to F16
            'gguf-schnell-q2k': 'flux1-schnell-Q2_K.gguf',
            'gguf-schnell-q3ks': 'flux1-schnell-Q3_K_S.gguf',
            'gguf-schnell-q4-0': 'flux1-schnell-Q4_0.gguf',
            'gguf-schnell-q4-1': 'flux1-schnell-Q4_1.gguf',
            'gguf-schnell-q4ks': 'flux1-schnell-Q4_K_S.gguf',
            'gguf-schnell-q5-0': 'flux1-schnell-Q5_0.gguf',
            'gguf-schnell-q5-1': 'flux1-schnell-Q5_1.gguf',
            'gguf-schnell-q5ks': 'flux1-schnell-Q5_K_S.gguf',
            'gguf-schnell-q6k': 'flux1-schnell-Q6_K.gguf',
            'gguf-schnell-q8': 'flux1-schnell-Q8_0.gguf',
            'gguf-schnell-f16': 'flux1-schnell-F16.gguf',
            
            # Stable Diffusion 3.5 Large variants
            'gguf-large': 'sd3.5_large-F16.gguf',  # Default to F16
            'gguf-large-q4-0': 'sd3.5_large-Q4_0.gguf',
            'gguf-large-q4-1': 'sd3.5_large-Q4_1.gguf',
            'gguf-large-q5-0': 'sd3.5_large-Q5_0.gguf',
            'gguf-large-q5-1': 'sd3.5_large-Q5_1.gguf',
            'gguf-large-q8-0': 'sd3.5_large-Q8_0.gguf',
            'gguf-large-f16': 'sd3.5_large-F16.gguf',
            
            # Stable Diffusion 3.5 Large Turbo variants
            'gguf-large-turbo': 'sd3.5_large_turbo.gguf',  # Default to standard format
            'gguf-large-turbo-q4-0': 'sd3.5_large_turbo-Q4_0.gguf',
            'gguf-large-turbo-q4-1': 'sd3.5_large_turbo-Q4_1.gguf',
            'gguf-large-turbo-q5-0': 'sd3.5_large_turbo-Q5_0.gguf',
            'gguf-large-turbo-q5-1': 'sd3.5_large_turbo-Q5_1.gguf',
            'gguf-large-turbo-q8-0': 'sd3.5_large_turbo-Q8_0.gguf',
            'gguf-large-turbo-f16': 'sd3.5_large_turbo-F16.gguf',
            
            # Other model variants
            'gguf-medium': 'sd3.5-medium-F16.gguf',
            'gguf-sd3-medium': 'sd3-medium-F16.gguf',
            'gguf-lite': 'flux-lite-8b-F16.gguf',
            'gguf-distilled': 'flux-dev-de-distill-F16.gguf',
            'gguf-fill': 'flux-fill-dev-F16.gguf',
            'gguf-full': 'hidream-i1-full-F16.gguf',
            'gguf-dev': 'hidream-i1-dev-F16.gguf',
            'gguf-fast': 'hidream-i1-fast-F16.gguf',
            'gguf-i2v': 'ltx-video-i2v-F16.gguf',
            'gguf-2b': 'ltx-video-2b-F16.gguf',
            'gguf-t2v': 'hunyuan-video-t2v-F16.gguf',
            
            'gguf': 'flux1-dev-Q4_K_S.gguf',  # Default to Q4_K_S
        }
        
        filename = variant_mapping.get(variant.lower())
        if filename:
            gguf_file = model_dir / filename
            if gguf_file.exists():
                return gguf_file
        
        # Fallback: search for any .gguf file
        gguf_files = list(model_dir.glob('*.gguf'))
        if gguf_files:
            return gguf_files[0]  # Return first found
            
        return None
    
    def get_additional_model_files(self, model_dir: Path) -> Dict[str, Optional[Path]]:
        """Find additional model files required for FLUX GGUF inference"""
        files = {
            'vae': None,
            'clip_l': None,
            't5xxl': None
        }
        
        # Common file patterns for FLUX models
        vae_patterns = ['ae.safetensors', 'vae.safetensors', 'flux_vae.safetensors']
        clip_l_patterns = ['clip_l.safetensors', 'text_encoder.safetensors']
        t5xxl_patterns = ['t5xxl_fp16.safetensors', 't5xxl.safetensors', 't5_encoder.safetensors']
        
        # Search for VAE
        for pattern in vae_patterns:
            vae_file = model_dir / pattern
            if vae_file.exists():
                files['vae'] = vae_file
                break
        
        # Search for CLIP-L
        for pattern in clip_l_patterns:
            clip_file = model_dir / pattern
            if clip_file.exists():
                files['clip_l'] = clip_file
                break
                
        # Search for T5XXL
        for pattern in t5xxl_patterns:
            t5_file = model_dir / pattern
            if t5_file.exists():
                files['t5xxl'] = t5_file
                break
        
        return files
    
    def load_model(self, model_config: Dict[str, Any], model_name: str = None, model_path: Path = None) -> bool:
        """Load GGUF model using stable-diffusion.cpp"""
        # Extract parameters from model_config if not provided separately
        if model_name is None:
            model_name = model_config.get('name', 'unknown')
        if model_path is None:
            model_path = Path(model_config.get('path', ''))
        
        logger.info(f"Loading GGUF model: {model_name}")
        
        try:
            # Find the GGUF file
            gguf_files = list(model_path.glob("*.gguf"))
            if not gguf_files:
                logger.error(f"No GGUF files found in {model_path}")
                return False
            
            gguf_file = gguf_files[0]  # Use the first GGUF file found
            logger.info(f"Using GGUF file: {gguf_file}")
            
            # Download required components
            components = self.download_required_components(model_path)
            
            # Detect model type for appropriate validation
            is_sd35 = any(pattern in model_name.lower() for pattern in ['3.5', 'sd3.5', 'stable-diffusion-3-5'])
            
            # Validate components based on model type
            if is_sd35:
                # SD 3.5 models need VAE, CLIP-L, CLIP-G, and T5XXL
                required_components = ['vae', 'clip_l', 'clip_g', 't5xxl']
                missing_components = [name for name in required_components if not components.get(name)]
                if missing_components:
                    logger.error(f"Missing required SD 3.5 components: {missing_components}")
                    return False
            else:
                # FLUX models need VAE, CLIP-L, and T5XXL (no CLIP-G)
                required_components = ['vae', 'clip_l', 't5xxl']
                missing_components = [name for name in required_components if not components.get(name)]
                if missing_components:
                    logger.error(f"Missing required FLUX components: {missing_components}")
                    return False
            
            # Initialize the stable-diffusion.cpp model
            logger.info("Loading GGUF model with stable-diffusion.cpp...")
            
            if is_sd35:
                logger.info("Detected SD 3.5 model - using appropriate configuration")
                
                sd_params = {
                    'diffusion_model_path': str(gguf_file),
                    'n_threads': 4
                }
                
                if components['vae']:
                    sd_params['vae_path'] = str(components['vae'])
                if components['clip_l']:
                    sd_params['clip_l_path'] = str(components['clip_l'])
                if components['clip_g']:
                    sd_params['clip_g_path'] = str(components['clip_g'])
                if components['t5xxl']:
                    sd_params['t5xxl_path'] = str(components['t5xxl'])
                
                logger.info(f"Initializing SD 3.5 model with params: {sd_params}")
                self.stable_diffusion = StableDiffusion(**sd_params)
                
            else:
                # FLUX models use different parameter structure
                logger.info("Detected FLUX model - using CLIP-L and T5-XXL configuration")
                self.stable_diffusion = StableDiffusion(
                    diffusion_model_path=str(gguf_file),
                    vae_path=str(components['vae']),
                    clip_l_path=str(components['clip_l']),
                    t5xxl_path=str(components['t5xxl']),
                    vae_decode_only=True,
                    n_threads=-1
                )
            
            self.model_path = str(gguf_file)
            self.model_config = model_config
            self.loaded_model_name = model_name
            
            logger.info(f"Successfully loaded GGUF model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model {model_name}: {e}")
            if hasattr(self, 'stable_diffusion') and self.stable_diffusion:
                self.stable_diffusion = None
            return False
    
    def generate_image(self, prompt: str, **kwargs) -> Optional[Image.Image]:
        """Generate image using stable-diffusion.cpp FLUX inference"""
        if not self.stable_diffusion:
            logger.error("GGUF model not loaded")
            return None
        
        try:
            # Extract parameters with FLUX-optimized defaults
            # Support both parameter naming conventions for compatibility
            width = kwargs.get('width', 1024)
            height = kwargs.get('height', 1024)
            
            # Support both 'steps' and 'num_inference_steps' - ensure not None
            steps = kwargs.get('steps') or kwargs.get('num_inference_steps') or 20
            
            # Support both 'cfg_scale' and 'guidance_scale' - FLUX works best with low CFG - ensure not None
            cfg_scale = kwargs.get('cfg_scale') or kwargs.get('guidance_scale') or 1.0
            
            seed = kwargs.get('seed', 42)
            negative_prompt = kwargs.get('negative_prompt', "")
            
            # Allow custom sampler, with FLUX-optimized default
            sampler = kwargs.get('sampler', kwargs.get('sample_method', 'dpmpp2m'))
            
            # Validate sampler and provide fallback
            valid_samplers = ['euler_a', 'euler', 'heun', 'dpm2', 'dpmpp2s_a', 'dpmpp2m', 'dpmpp2mv2', 'ipndm', 'ipndm_v', 'lcm', 'ddim_trailing', 'tcd']
            if sampler not in valid_samplers:
                logger.warning(f"Invalid sampler '{sampler}', falling back to 'dpmpp2m'")
                sampler = 'dpmpp2m'
            
            # Ensure all values are proper types and not None
            steps = int(steps) if steps is not None else 20
            cfg_scale = float(cfg_scale) if cfg_scale is not None else 1.0
            width = int(width) if width is not None else 1024
            height = int(height) if height is not None else 1024
            seed = int(seed) if seed is not None else 42
            
            logger.info(f"Generating image: {width}x{height}, steps={steps}, cfg={cfg_scale}, sampler={sampler}, negative_prompt={negative_prompt}")
            
            # Log model quantization info for quality assessment
            if hasattr(self, 'model_path'):
                if 'Q2' in str(self.model_path):
                    logger.warning("Using Q2 quantization - expect lower quality. Consider Q4_K_S or higher for better results.")
                elif 'Q3' in str(self.model_path):
                    logger.info("Using Q3 quantization - moderate quality. Consider Q4_K_S or higher for better results.")
                elif 'Q4' in str(self.model_path):
                    logger.info("Using Q4 quantization - good balance of quality and size.")
                elif any(x in str(self.model_path) for x in ['Q5', 'Q6', 'Q8', 'F16']):
                    logger.info("Using high precision quantization - excellent quality expected.")
            
            # Generate image using stable-diffusion.cpp
            # According to the documentation, txt_to_img returns a list of PIL Images
            try:
                result = self.stable_diffusion.txt_to_img(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else "",
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    sample_method=sampler,
                    sample_steps=steps,
                    seed=seed
                )
                logger.info(f"txt_to_img returned: {type(result)}, length: {len(result) if result else 'None'}")
            except Exception as e:
                logger.error(f"txt_to_img call failed: {e}")
                return None
            
            if not result:
                logger.error("txt_to_img returned None")
                return None
                
            if not isinstance(result, list) or len(result) == 0:
                logger.error(f"txt_to_img returned unexpected format: {type(result)}")
                return None
            
            # Get the first PIL Image from the result list
            image = result[0]
            logger.info(f"Retrieved PIL Image: {type(image)}")
            
            # Verify it's a PIL Image
            if not hasattr(image, 'save'):
                logger.error(f"Result[0] is not a PIL Image: {type(image)}")
                return None
            
            # Optionally save a copy for debugging/history
            try:
                from ..config.settings import settings
                output_dir = settings.config_dir / "outputs"
                output_dir.mkdir(exist_ok=True)
                
                output_path = output_dir / f"gguf_output_{seed}.png"
                image.save(output_path)
                logger.info(f"Generated image also saved to: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save debug copy: {e}")
            
            # Return the PIL Image directly for API compatibility
            logger.info("Returning PIL Image for API use")
            return image
            
        except Exception as e:
            logger.error(f"Failed to generate image with GGUF model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def unload_model(self):
        """Unload the GGUF model"""
        if self.stable_diffusion:
            try:
                # stable-diffusion-cpp handles cleanup automatically
                self.stable_diffusion = None
                self.model_path = None
                self.model_config = None
                self.loaded_model_name = None
                logger.info("GGUF model unloaded")
            except Exception as e:
                logger.error(f"Error unloading GGUF model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.stable_diffusion:
            return {
                'gguf_available': GGUF_AVAILABLE,
                'loaded': False
            }
            
        return {
            'type': 'gguf',
            'variant': self.model_config.get('variant', 'unknown'),
            'path': str(self.model_path),
            'name': self.loaded_model_name,
            'loaded': True,
            'gguf_available': GGUF_AVAILABLE,
            'backend': 'stable-diffusion.cpp'
        }
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded"""
        return self.stable_diffusion is not None

    def get_gguf_download_patterns(self, variant: str) -> Dict[str, List[str]]:
        """Get file patterns for downloading specific GGUF variant
        
        Args:
            variant: Model variant (e.g., 'gguf-q4-1', 'gguf-q4ks')
            
        Returns:
            Dict with 'allow_patterns' and 'ignore_patterns' lists
        """
        # Map variant to specific GGUF file patterns
        variant_patterns = {
            # FLUX.1-dev variants
            'gguf-q2k': ['*Q2_K*.gguf'],
            'gguf-q3ks': ['*Q3_K_S*.gguf'], 
            'gguf-q4ks': ['*Q4_K_S*.gguf'],
            'gguf-q4-0': ['*Q4_0*.gguf'],
            'gguf-q4-1': ['*Q4_1*.gguf'],
            'gguf-q5ks': ['*Q5_K_S*.gguf'],
            'gguf-q5-0': ['*Q5_0*.gguf'],
            'gguf-q5-1': ['*Q5_1*.gguf'],
            'gguf-q6k': ['*Q6_K*.gguf'],
            'gguf-q8': ['*Q8_0*.gguf'],
            'gguf-q8-0': ['*Q8_0*.gguf'],  # Keep for backward compatibility
            'gguf-f16': ['*F16*.gguf'],
            
            # FLUX.1-schnell variants
            'gguf-schnell': ['*flux1-schnell*F16*.gguf'],
            'gguf-schnell-q2k': ['*flux1-schnell*Q2_K*.gguf'],
            'gguf-schnell-q3ks': ['*flux1-schnell*Q3_K_S*.gguf'],
            'gguf-schnell-q4-0': ['*flux1-schnell*Q4_0*.gguf'],
            'gguf-schnell-q4-1': ['*flux1-schnell*Q4_1*.gguf'],
            'gguf-schnell-q4ks': ['*flux1-schnell*Q4_K_S*.gguf'],
            'gguf-schnell-q5-0': ['*flux1-schnell*Q5_0*.gguf'],
            'gguf-schnell-q5-1': ['*flux1-schnell*Q5_1*.gguf'],
            'gguf-schnell-q5ks': ['*flux1-schnell*Q5_K_S*.gguf'],
            'gguf-schnell-q6k': ['*flux1-schnell*Q6_K*.gguf'],
            'gguf-schnell-q8': ['*flux1-schnell*Q8_0*.gguf'],
            'gguf-schnell-f16': ['*flux1-schnell*F16*.gguf'],
            
            # Stable Diffusion 3.5 Large variants
            'gguf-large': ['*sd3.5_large-F16*.gguf'],
            'gguf-large-q4-0': ['*sd3.5_large-Q4_0*.gguf'],
            'gguf-large-q4-1': ['*sd3.5_large-Q4_1*.gguf'],
            'gguf-large-q5-0': ['*sd3.5_large-Q5_0*.gguf'],
            'gguf-large-q5-1': ['*sd3.5_large-Q5_1*.gguf'],
            'gguf-large-q8-0': ['*sd3.5_large-Q8_0*.gguf'],
            'gguf-large-f16': ['*sd3.5_large-F16*.gguf'],
            
            # Stable Diffusion 3.5 Large Turbo variants
            'gguf-large-turbo': ['*sd3.5_large_turbo*F16*.gguf'],
            'gguf-large-turbo-q4-0': ['*sd3.5_large_turbo*Q4_0*.gguf'],
            'gguf-large-turbo-q4-1': ['*sd3.5_large_turbo*Q4_1*.gguf'],
            'gguf-large-turbo-q5-0': ['*sd3.5_large_turbo*Q5_0*.gguf'],
            'gguf-large-turbo-q5-1': ['*sd3.5_large_turbo*Q5_1*.gguf'],
            'gguf-large-turbo-q8-0': ['*sd3.5_large_turbo*Q8_0*.gguf'],
            'gguf-large-turbo-f16': ['*sd3.5_large_turbo*F16*.gguf'],
            
            # Other model variants
            'gguf-medium': ['*sd3.5-medium*.gguf'],
            'gguf-sd3-medium': ['*sd3-medium*.gguf'],
            'gguf-lite': ['*flux-lite-8b*.gguf'],
            'gguf-distilled': ['*flux-dev-de-distill*.gguf'],
            'gguf-fill': ['*flux-fill-dev*.gguf'],
            'gguf-full': ['*hidream-i1-full*.gguf'],
            'gguf-dev': ['*hidream-i1-dev*.gguf'],
            'gguf-fast': ['*hidream-i1-fast*.gguf'],
            'gguf-i2v': ['*ltx-video-i2v*.gguf', '*hunyuan-video-i2v*.gguf'],
            'gguf-2b': ['*ltx-video-2b*.gguf'],
            'gguf-t2v': ['*hunyuan-video-t2v*.gguf'],
        }
        
        # Get the specific GGUF file pattern for this variant
        gguf_pattern = variant_patterns.get(variant, ['*.gguf'])
        
        # Essential files to download
        essential_files = [
            # Configuration and metadata
            'model_index.json',
            'README.md',
            'LICENSE*',
            '.gitattributes',
            'config.json',
        ]
        
        # Include the specific GGUF model file
        allow_patterns = essential_files + gguf_pattern
        
        # Create ignore patterns based on variant name (not pattern content)
        # This prevents conflicts between allow and ignore patterns
        ignore_patterns = []
        
        # Determine model family from variant name
        if variant.startswith('gguf-schnell') or 'schnell' in variant:
            # FLUX.1-schnell variants - ignore other model types
            ignore_patterns = [
                '*flux1-dev*.gguf',     # Ignore FLUX.1-dev
                '*sd3.5*.gguf',         # Ignore SD 3.5
                '*ltx-video*.gguf',     # Ignore video models
                '*hidream*.gguf',       # Ignore HiDream
                '*hunyuan*.gguf'        # Ignore Hunyuan
            ]
            # Ignore other schnell quantizations except the one we want
            for other_variant, other_patterns in variant_patterns.items():
                if (other_variant.startswith('gguf-schnell') and 
                    other_variant != variant and 
                    other_variant != 'gguf'):
                    # Only ignore if it doesn't conflict with our allow patterns
                    for pattern in other_patterns:
                        if pattern not in gguf_pattern:
                            ignore_patterns.append(pattern)
                    
        elif (variant.startswith('gguf-large-turbo') or 
              'large-turbo' in variant or 
              variant.startswith('gguf-large') or
              'sd3.5' in variant or
              'stable-diffusion-3' in variant):
            # SD 3.5 variants - ignore other model types  
            ignore_patterns = [
                '*flux1-dev*.gguf',     # Ignore FLUX.1-dev
                '*flux1-schnell*.gguf', # Ignore FLUX.1-schnell
                '*ltx-video*.gguf',     # Ignore video models
                '*hidream*.gguf',       # Ignore HiDream  
                '*hunyuan*.gguf'        # Ignore Hunyuan
            ]
            # Ignore other SD 3.5 quantizations except the one we want
            for other_variant, other_patterns in variant_patterns.items():
                if (('large' in other_variant or 'sd3.5' in other_variant or 'stable-diffusion-3' in other_variant) and 
                    other_variant != variant and 
                    other_variant != 'gguf'):
                    # Only ignore if it doesn't conflict with our allow patterns
                    for pattern in other_patterns:
                        if pattern not in gguf_pattern:
                            ignore_patterns.append(pattern)
                    
        elif ('video' in variant or 
              'i2v' in variant or 
              't2v' in variant or 
              '2b' in variant):
            # Video model variants
            ignore_patterns = [
                '*flux1-dev*.gguf',
                '*flux1-schnell*.gguf',
                '*sd3.5*.gguf'
            ]
            
        elif ('hidream' in variant or 
              'full' in variant or 
              'fast' in variant):
            # HiDream variants  
            ignore_patterns = [
                '*flux1-dev*.gguf',
                '*flux1-schnell*.gguf', 
                '*sd3.5*.gguf',
                '*ltx-video*.gguf',
                '*hunyuan*.gguf'
            ]
            
        else:
            # FLUX.1-dev variants (default case) - ignore other model types
            ignore_patterns = [
                '*flux1-schnell*.gguf', # Ignore FLUX.1-schnell
                '*sd3.5*.gguf',         # Ignore SD 3.5
                '*ltx-video*.gguf',     # Ignore video models
                '*hidream*.gguf',       # Ignore HiDream
                '*hunyuan*.gguf'        # Ignore Hunyuan
            ]
            # Ignore other FLUX.1-dev quantizations except the one we want
            for other_variant, other_patterns in variant_patterns.items():
                if (not other_variant.startswith('gguf-schnell') and 
                    not 'large' in other_variant and
                    not 'sd3.5' in other_variant and
                    not 'video' in other_variant and
                    not 'hidream' in other_variant and
                    other_variant != variant and 
                    other_variant != 'gguf'):
                    # Only ignore if it doesn't conflict with our allow patterns
                    for pattern in other_patterns:
                        if pattern not in gguf_pattern:
                            ignore_patterns.append(pattern)
        
        return {
            'allow_patterns': allow_patterns,
            'ignore_patterns': ignore_patterns
        }
    
    def _get_model_family(self, pattern: str) -> str:
        """Extract model family from a pattern (e.g., flux1-dev, flux1-schnell, sd3.5-large)"""
        if 'flux1-dev' in pattern:
            return 'flux1-dev'
        elif 'flux1-schnell' in pattern:
            return 'flux1-schnell'
        elif 'sd3.5-large-turbo' in pattern:
            return 'sd3.5-large-turbo'
        elif 'sd3.5-large' in pattern:
            return 'sd3.5-large'
        elif 'sd3.5' in pattern:
            return 'sd3.5'
        else:
            return pattern.split('*')[1].split('*')[0] if '*' in pattern else pattern
    
    def download_required_components(self, model_path: Path) -> Dict[str, Optional[Path]]:
        """Download or locate required VAE, CLIP-L, and T5XXL components
        
        For different model types:
        - FLUX GGUF models need: ae.safetensors (VAE), clip_l.safetensors, t5xxl_fp16.safetensors
        - SD 3.5 models need: different text encoders and VAE
        """
        from ..utils.download_utils import robust_snapshot_download
        from ..config.settings import settings
        
        components = {
            'vae': None,
            'clip_l': None, 
            'clip_g': None,  # Needed for SD 3.5 models
            't5xxl': None
        }
        
        # Detect model type based on model path or name
        model_name = model_path.name.lower()
        is_sd35 = any(pattern in model_name for pattern in ['3.5', 'sd3.5', 'stable-diffusion-3-5'])
        is_flux = any(x in model_name for x in ['flux', 'flux1'])
        
        logger.info(f"Downloading required components for model type: {'SD3.5' if is_sd35 else 'FLUX' if is_flux else 'Unknown'}")
        
        try:
            if is_sd35:
                # SD 3.5 models - use SD 3.5 specific components
                logger.info("Downloading SD 3.5 components...")
                
                # Download SD 3.5 VAE
                vae_dir = model_path.parent / "sd35_vae"
                if not (vae_dir / "vae.safetensors").exists():
                    logger.info("Downloading SD 3.5 VAE...")
                    robust_snapshot_download(
                        repo_id="stabilityai/stable-diffusion-3.5-large",
                        local_dir=str(vae_dir),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=['vae/diffusion_pytorch_model.safetensors'],
                        max_retries=3
                    )
                    # Move to expected location if needed
                    vae_source = vae_dir / "vae" / "diffusion_pytorch_model.safetensors"
                    vae_target = vae_dir / "vae.safetensors"
                    if vae_source.exists() and not vae_target.exists():
                        vae_source.rename(vae_target)
                
                vae_path = vae_dir / "vae.safetensors"
                if vae_path.exists():
                    components['vae'] = vae_path
                    logger.info(f"SD 3.5 VAE found at: {vae_path}")
                
                # Download SD 3.5 text encoders
                text_encoders_dir = model_path.parent / "sd35_text_encoders"
                
                # Download CLIP-L for SD 3.5
                if not (text_encoders_dir / "clip_l.safetensors").exists():
                    logger.info("Downloading SD 3.5 CLIP-L text encoder...")
                    robust_snapshot_download(
                        repo_id="stabilityai/stable-diffusion-3.5-large",
                        local_dir=str(text_encoders_dir),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=['text_encoders/clip_l.safetensors'],
                        max_retries=3
                    )
                    # Move to expected location if needed
                    clip_source = text_encoders_dir / "text_encoders" / "clip_l.safetensors"
                    clip_target = text_encoders_dir / "clip_l.safetensors"
                    if clip_source.exists() and not clip_target.exists():
                        clip_source.rename(clip_target)
                
                clip_l_path = text_encoders_dir / "clip_l.safetensors"
                if clip_l_path.exists():
                    components['clip_l'] = clip_l_path
                    logger.info(f"SD 3.5 CLIP-L found at: {clip_l_path}")
                
                # Download CLIP-G for SD 3.5  
                if not (text_encoders_dir / "clip_g.safetensors").exists():
                    logger.info("Downloading SD 3.5 CLIP-G text encoder...")
                    robust_snapshot_download(
                        repo_id="stabilityai/stable-diffusion-3.5-large",
                        local_dir=str(text_encoders_dir),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=['text_encoders/clip_g.safetensors'],
                        max_retries=3
                    )
                    # Move to expected location if needed
                    clipg_source = text_encoders_dir / "text_encoders" / "clip_g.safetensors"
                    clipg_target = text_encoders_dir / "clip_g.safetensors"
                    if clipg_source.exists() and not clipg_target.exists():
                        clipg_source.rename(clipg_target)
                
                clip_g_path = text_encoders_dir / "clip_g.safetensors"
                if clip_g_path.exists():
                    components['clip_g'] = clip_g_path
                    logger.info(f"SD 3.5 CLIP-G found at: {clip_g_path}")
                
                # Download T5XXL for SD 3.5
                if not (text_encoders_dir / "t5xxl_fp16.safetensors").exists():
                    logger.info("Downloading SD 3.5 T5XXL text encoder...")
                    robust_snapshot_download(
                        repo_id="stabilityai/stable-diffusion-3.5-large",
                        local_dir=str(text_encoders_dir),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=['text_encoders/t5xxl_fp16.safetensors'],
                        max_retries=3
                    )
                    # Move to expected location if needed
                    t5_source = text_encoders_dir / "text_encoders" / "t5xxl_fp16.safetensors"
                    t5_target = text_encoders_dir / "t5xxl_fp16.safetensors"
                    if t5_source.exists() and not t5_target.exists():
                        t5_source.rename(t5_target)
                
                t5xxl_path = text_encoders_dir / "t5xxl_fp16.safetensors"
                if t5xxl_path.exists():
                    components['t5xxl'] = t5xxl_path
                    logger.info(f"SD 3.5 T5XXL found at: {t5xxl_path}")
                    
            else:
                # FLUX models (default) - use FLUX specific components
                logger.info("Downloading FLUX components...")
                
                # Download VAE from official FLUX repository
                vae_dir = model_path.parent / "flux_vae"
                if not (vae_dir / "ae.safetensors").exists():
                    logger.info("Downloading FLUX VAE...")
                    robust_snapshot_download(
                        repo_id="black-forest-labs/FLUX.1-dev",
                        local_dir=str(vae_dir),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=['ae.safetensors'],
                        max_retries=3
                    )
                
                vae_path = vae_dir / "ae.safetensors"
                if vae_path.exists():
                    components['vae'] = vae_path
                    logger.info(f"FLUX VAE found at: {vae_path}")
                
                # Download text encoders
                text_encoders_dir = model_path.parent / "flux_text_encoders"
                
                # Download CLIP-L
                if not (text_encoders_dir / "clip_l.safetensors").exists():
                    logger.info("Downloading FLUX CLIP-L text encoder...")
                    robust_snapshot_download(
                        repo_id="comfyanonymous/flux_text_encoders",
                        local_dir=str(text_encoders_dir),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=['clip_l.safetensors'],
                        max_retries=3
                    )
                
                clip_l_path = text_encoders_dir / "clip_l.safetensors"
                if clip_l_path.exists():
                    components['clip_l'] = clip_l_path
                    logger.info(f"FLUX CLIP-L found at: {clip_l_path}")
                
                # Download T5XXL  
                if not (text_encoders_dir / "t5xxl_fp16.safetensors").exists():
                    logger.info("Downloading FLUX T5XXL text encoder...")
                    robust_snapshot_download(
                        repo_id="comfyanonymous/flux_text_encoders", 
                        local_dir=str(text_encoders_dir),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=['t5xxl_fp16.safetensors'],
                        max_retries=3
                    )
                
                t5xxl_path = text_encoders_dir / "t5xxl_fp16.safetensors"
                if t5xxl_path.exists():
                    components['t5xxl'] = t5xxl_path
                    logger.info(f"FLUX T5XXL found at: {t5xxl_path}")
                
        except Exception as e:
            logger.error(f"Failed to download components: {e}")
            
        return components


# Global GGUF loader instance
gguf_loader = GGUFModelLoader() 