import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import hashlib
from huggingface_hub import login
from ..config.settings import settings, ModelConfig
from ..utils.download_utils import robust_snapshot_download, robust_file_download

logger = logging.getLogger(__name__)

class ModelManager:
    """Model manager"""
    
    def __init__(self):
        self.loaded_model: Optional[object] = None
        self.current_model_name: Optional[str] = None
        
        # Predefined model registry
        self.model_registry = {
            "stable-diffusion-3.5-medium": {
                "repo_id": "stabilityai/stable-diffusion-3.5-medium",
                "model_type": "sd3",
                "variant": "fp16",
                "components": {
                    "lora": {
                        "repo_id": "tensorart/stable-diffusion-3.5-medium-turbo",
                        "filename": "lora_sd3.5m_turbo_8steps.safetensors"
                    }
                },
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5
                }
            },
            "stable-diffusion-xl-base": {
                "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "model_type": "sdxl",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                }
            },
            "stable-diffusion-1.5": {
                "repo_id": "runwayml/stable-diffusion-v1-5",
                "model_type": "sd15",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                }
            }
        }
    
    def list_available_models(self) -> List[str]:
        """List all available models"""
        return list(self.model_registry.keys())
    
    def list_installed_models(self) -> List[str]:
        """List installed models"""
        return list(settings.models.keys())
    
    def is_model_installed(self, model_name: str) -> bool:
        """Check if model is installed"""
        return model_name in settings.models
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information"""
        if model_name in self.model_registry:
            info = self.model_registry[model_name].copy()
            info['installed'] = self.is_model_installed(model_name)
            if info['installed']:
                config = settings.models[model_name]
                info['local_path'] = config.path
                info['size'] = self._get_model_size(config.path)
            return info
        return None
    
    def _get_model_size(self, model_path: str) -> str:
        """Get model size"""
        try:
            path = Path(model_path)
            if path.is_file():
                size = path.stat().st_size
            else:
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            # Convert to human readable format
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} PB"
        except Exception:
            return "Unknown"
    
    def pull_model(self, model_name: str, force: bool = False, progress_callback=None) -> bool:
        """Download model using robust download utilities"""
        if not force and self.is_model_installed(model_name):
            logger.info(f"Model {model_name} already exists")
            if progress_callback:
                progress_callback(f"Model {model_name} already installed")
            return True
        
        if model_name not in self.model_registry:
            logger.error(f"Unknown model: {model_name}")
            if progress_callback:
                progress_callback(f"Error: Unknown model {model_name}")
            return False
        
        model_info = self.model_registry[model_name]
        model_path = settings.get_model_path(model_name)
        
        try:
            # Ensure HuggingFace token is set
            if settings.hf_token:
                login(token=settings.hf_token)
            
            logger.info(f"Downloading model: {model_name}")
            if progress_callback:
                progress_callback(f"Starting download of {model_name}")
            
            # Download main model using robust downloader
            robust_snapshot_download(
                repo_id=model_info["repo_id"],
                local_dir=str(model_path),
                cache_dir=str(settings.cache_dir),
                max_retries=3,
                initial_workers=2,
                force_download=force,
                progress_callback=progress_callback
            )
            
            # Download components (such as LoRA)
            if "components" in model_info:
                components_path = model_path / "components"
                components_path.mkdir(exist_ok=True)
                
                for comp_name, comp_info in model_info["components"].items():
                    comp_path = components_path / comp_name
                    comp_path.mkdir(exist_ok=True)
                    
                    if progress_callback:
                        progress_callback(f"Downloading component: {comp_name}")
                    
                    if "filename" in comp_info:
                        # Download single file using robust downloader
                        robust_file_download(
                            repo_id=comp_info["repo_id"],
                            filename=comp_info["filename"],
                            local_dir=str(comp_path),
                            cache_dir=str(settings.cache_dir),
                            max_retries=3,
                            progress_callback=progress_callback
                        )
                    else:
                        # Download entire repository using robust downloader
                        robust_snapshot_download(
                            repo_id=comp_info["repo_id"],
                            local_dir=str(comp_path),
                            cache_dir=str(settings.cache_dir),
                            max_retries=3,
                            initial_workers=1,  # Use fewer workers for components
                            force_download=force,
                            progress_callback=progress_callback
                        )
            
            # Add to configuration
            model_config = ModelConfig(
                name=model_name,
                path=str(model_path),
                model_type=model_info["model_type"],
                variant=model_info.get("variant"),
                components=model_info.get("components"),
                parameters=model_info.get("parameters")
            )
            
            settings.add_model(model_config)
            logger.info(f"Model {model_name} download completed")
            if progress_callback:
                progress_callback(f"✅ {model_name} download completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            if progress_callback:
                progress_callback(f"❌ Download failed: {str(e)}")
            
            # Clean up failed download
            if model_path.exists():
                try:
                    shutil.rmtree(model_path)
                    logger.info(f"Cleaned up failed download directory: {model_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up directory {model_path}: {cleanup_error}")
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove model"""
        if not self.is_model_installed(model_name):
            logger.error(f"Model {model_name} is not installed")
            return False
        
        try:
            # If currently using this model, unload it first
            if self.current_model_name == model_name:
                self.unload_model()
            
            # Delete model files
            model_config = settings.models[model_name]
            model_path = Path(model_config.path)
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Remove from configuration
            settings.remove_model(model_name)
            
            logger.info(f"Model {model_name} has been removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model: {e}")
            return False
    
    def load_model(self, model_name: str) -> bool:
        """Load model into memory"""
        if not self.is_model_installed(model_name):
            logger.error(f"Model {model_name} is not installed")
            return False
        
        # If the same model is already loaded, return directly
        if self.current_model_name == model_name:
            logger.info(f"Model {model_name} is already loaded")
            return True
        
        # Unload current model
        if self.loaded_model is not None:
            self.unload_model()
        
        try:
            from ..inference.engine import InferenceEngine
            
            model_config = settings.models[model_name]
            engine = InferenceEngine()
            
            if engine.load_model(model_config):
                self.loaded_model = engine
                self.current_model_name = model_name
                settings.set_current_model(model_name)
                logger.info(f"Model {model_name} loaded successfully")
                return True
            else:
                logger.error(f"Model {model_name} failed to load")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def unload_model(self):
        """Unload current model"""
        if self.loaded_model is not None:
            try:
                self.loaded_model.unload()
                logger.info(f"Model {self.current_model_name} unloaded")
            except Exception as e:
                logger.error(f"Failed to unload model: {e}")
            finally:
                self.loaded_model = None
                self.current_model_name = None
        
        # Also clear the persisted state
        settings.current_model = None
        settings.save_config()
    
    def get_current_model(self) -> Optional[str]:
        """Get current loaded model name"""
        # First check in-memory state
        if self.current_model_name:
            return self.current_model_name
        # Then check persisted state
        return settings.current_model
    
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded in memory"""
        # Only check in-memory state - a model is truly loaded only if it's in memory
        return self.loaded_model is not None
    
    def has_current_model(self) -> bool:
        """Check if there's a current model set (may not be loaded in memory)"""
        return settings.current_model is not None
    
    def is_server_running(self) -> bool:
        """Check if the server is actually running"""
        try:
            import requests
            host = settings.server.host
            port = settings.server.port
            response = requests.get(f"http://{host}:{port}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False

# Global model manager instance
model_manager = ModelManager() 