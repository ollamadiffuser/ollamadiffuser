import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline
)
from PIL import Image
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Inference engine responsible for actual image generation"""
    
    def __init__(self):
        self.pipeline = None
        self.model_config: Optional[ModelConfig] = None
        self.device = None
        self.tokenizer = None
        self.max_token_limit = 77
        
    def _get_device(self) -> str:
        """Automatically detect available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    
    def _get_pipeline_class(self, model_type: str):
        """Get corresponding pipeline class based on model type"""
        pipeline_map = {
            "sd15": StableDiffusionPipeline,
            "sdxl": StableDiffusionXLPipeline,
            "sd3": StableDiffusion3Pipeline
        }
        return pipeline_map.get(model_type)
    
    def load_model(self, model_config: ModelConfig) -> bool:
        """Load model"""
        try:
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")
            
            # Get corresponding pipeline class
            pipeline_class = self._get_pipeline_class(model_config.model_type)
            if not pipeline_class:
                logger.error(f"Unsupported model type: {model_config.model_type}")
                return False
            
            # Set loading parameters
            load_kwargs = {}
            if model_config.variant == "fp16":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["variant"] = "fp16"
            
            # Load pipeline
            logger.info(f"Loading model: {model_config.name}")
            self.pipeline = pipeline_class.from_pretrained(
                model_config.path,
                **load_kwargs
            ).to(self.device)
            
            # Load LoRA and other components
            if model_config.components and "lora" in model_config.components:
                self._load_lora(model_config)
            
            # Apply optimizations
            self._apply_optimizations()
            
            # Set tokenizer
            if hasattr(self.pipeline, 'tokenizer'):
                self.tokenizer = self.pipeline.tokenizer
            
            self.model_config = model_config
            logger.info(f"Model {model_config.name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_lora(self, model_config: ModelConfig):
        """Load LoRA weights"""
        try:
            lora_config = model_config.components["lora"]
            components_path = Path(model_config.path) / "components" / "lora"
            
            if "filename" in lora_config:
                # Load from local file
                lora_path = components_path / lora_config["filename"]
                if lora_path.exists():
                    self.pipeline.load_lora_weights(str(components_path), weight_name=lora_config["filename"])
                    self.pipeline.fuse_lora()
                    logger.info("LoRA weights loaded successfully")
            else:
                # Load from directory
                if components_path.exists():
                    self.pipeline.load_lora_weights(str(components_path))
                    self.pipeline.fuse_lora()
                    logger.info("LoRA weights loaded successfully")
                    
        except Exception as e:
            logger.warning(f"Failed to load LoRA weights: {e}")
    
    def _apply_optimizations(self):
        """Apply performance optimizations"""
        try:
            # Enable attention slicing to save memory
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            # Enable torch.compile acceleration (MPS not supported)
            if hasattr(torch, 'compile') and self.device != "mps":
                if hasattr(self.pipeline, 'unet'):
                    self.pipeline.unet = torch.compile(
                        self.pipeline.unet, 
                        mode="reduce-overhead", 
                        fullgraph=True
                    )
                    logger.info("torch.compile optimization enabled")
                    
        except Exception as e:
            logger.warning(f"Failed to apply optimization settings: {e}")
    
    def truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to fit CLIP token limit"""
        if not prompt or not self.tokenizer:
            return prompt
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        
        # Check if truncation is needed
        if len(tokens) <= self.max_token_limit:
            return prompt
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:self.max_token_limit]
        truncated_prompt = self.tokenizer.decode(truncated_tokens)
        
        logger.warning(f"Prompt truncated: {len(tokens)} -> {len(truncated_tokens)} tokens")
        return truncated_prompt
    
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
                      num_inference_steps: Optional[int] = None,
                      guidance_scale: Optional[float] = None,
                      width: int = 1024,
                      height: int = 1024,
                      **kwargs) -> Image.Image:
        """Generate image"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        
        # Use model default parameters
        if num_inference_steps is None:
            num_inference_steps = self.model_config.parameters.get("num_inference_steps", 28)
        
        if guidance_scale is None:
            guidance_scale = self.model_config.parameters.get("guidance_scale", 3.5)
        
        # Truncate prompts
        truncated_prompt = self.truncate_prompt(prompt)
        truncated_negative_prompt = self.truncate_prompt(negative_prompt)
        
        try:
            logger.info(f"Starting image generation: {truncated_prompt[:50]}...")
            
            # Generation parameters
            generation_kwargs = {
                "prompt": truncated_prompt,
                "negative_prompt": truncated_negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                **kwargs
            }
            
            # Add size parameters for SDXL and SD3
            if self.model_config.model_type in ["sdxl", "sd3"]:
                generation_kwargs.update({
                    "width": width,
                    "height": height
                })
            
            # Generate image
            output = self.pipeline(**generation_kwargs)
            
            image = output.images[0]
            logger.info("Image generation completed")
            return image
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            # Return error image
            return self._create_error_image(str(e), truncated_prompt)
    
    def _create_error_image(self, error_msg: str, prompt: str) -> Image.Image:
        """Create error message image"""
        from PIL import ImageDraw, ImageFont
        
        # Create white background image
        img = Image.new('RGB', (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw error information
        try:
            # Try to use system font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text
        draw.text((10, 10), f"Error: {error_msg}", fill=(255, 0, 0), font=font)
        draw.text((10, 30), f"Prompt: {prompt[:50]}...", fill=(0, 0, 0), font=font)
        
        return img
    
    def unload(self):
        """Unload model and free GPU memory"""
        if self.pipeline:
            # Move to CPU to free GPU memory
            self.pipeline = self.pipeline.to("cpu")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete pipeline
            del self.pipeline
            self.pipeline = None
            self.model_config = None
            self.tokenizer = None
            
            logger.info("Model unloaded")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipeline is not None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get current loaded model information"""
        if not self.model_config:
            return None
        
        return {
            "name": self.model_config.name,
            "type": self.model_config.model_type,
            "device": self.device,
            "variant": self.model_config.variant,
            "parameters": self.model_config.parameters
        } 