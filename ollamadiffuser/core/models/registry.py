"""
Dynamic Model Registry - Similar to Ollama's approach
Fetches model information from external sources with local fallbacks
"""

import json
import logging
import requests
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Dynamic model registry that can fetch from external sources
    Similar to how Ollama manages their model library
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".ollamadiffuser" / "registry"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        # Cache settings from config
        self.cache_duration = timedelta(hours=self.config.get('cache_duration_hours', 24))
        self.registry_cache_file = self.cache_dir / "models.json"
        self.last_update_file = self.cache_dir / "last_update.txt"
        
        # Registry sources from config
        self.registry_sources = self.config.get('sources', [])
        
        # Local models (built-in fallback)
        self._builtin_models = self._load_builtin_models()
        
        # Cached models
        self._cached_models = {}
        self._load_cache()
    
    def _load_config(self):
        """Load registry configuration from YAML file"""
        try:
            # Try to find config file
            config_paths = [
                Path(__file__).parent.parent.parent / "config" / "registry.yaml",
                Path.home() / ".ollamadiffuser" / "registry.yaml",
                Path("/etc/ollamadiffuser/registry.yaml")
            ]
            
            config = {}
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    logger.debug(f"Loaded config from {config_path}")
                    break
            
            # Use registry section if it exists
            self.config = config.get('registry', {})
            
            # Set defaults if no config found
            if not self.config:
                logger.warning("No registry config found, using defaults")
                self.config = {
                    'cache_duration_hours': 24,
                    'sources': [
                        {
                            "name": "builtin",
                            "url": None,
                            "timeout": 10,
                            "enabled": True,
                            "description": "Built-in models only"
                        }
                    ]
                }
                
        except Exception as e:
            logger.warning(f"Failed to load registry config: {e}")
            self.config = {'cache_duration_hours': 24, 'sources': []}
    
    def _load_builtin_models(self) -> Dict[str, Any]:
        """Load built-in model definitions as fallback"""
        return {
            # FLUX.1 models
            "flux.1-dev": {
                "name": "flux.1-dev",
                "repo_id": "black-forest-labs/FLUX.1-dev",
                "model_type": "flux",
                "description": "High-quality text-to-image model from Black Forest Labs",
                "license": {"type": "Non-commercial", "commercial_use": False},
                "size_gb": 23.8,
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32
                },
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "max_sequence_length": 512
                },
                "tags": ["flux", "high-quality", "non-commercial"],
                "downloads": 250000,
                "updated": "2024-12-01"
            },
            
            "flux.1-schnell": {
                "name": "flux.1-schnell",
                "repo_id": "black-forest-labs/FLUX.1-schnell",
                "model_type": "flux",
                "description": "Fast text-to-image model optimized for speed",
                "license": {"type": "Apache 2.0", "commercial_use": True},
                "size_gb": 23.8,
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32
                },
                "parameters": {
                    "num_inference_steps": 4,
                    "guidance_scale": 0.0,
                    "max_sequence_length": 512
                },
                "tags": ["flux", "fast", "commercial", "apache"],
                "downloads": 180000,
                "updated": "2024-12-01"
            },
            
            # GGUF variants - generate dynamically
            **self._generate_gguf_variants()
        }
    
    def _generate_gguf_variants(self) -> Dict[str, Any]:
        """Generate GGUF model variants dynamically"""
        base_gguf = {
            "repo_id": "city96/FLUX.1-dev-gguf",
            "model_type": "flux_gguf",
            "description": "Quantized FLUX.1-dev model for efficient inference",
            "license": {"type": "Non-commercial", "commercial_use": False},
            "tags": ["flux", "gguf", "quantized", "efficient"],
            "updated": "2024-12-01"
        }
        
        variants = {
            "q2_k": {"size_gb": 4.03, "vram_gb": 4, "description": "Ultra-light quantization"},
            "q3_k_s": {"size_gb": 5.23, "vram_gb": 5, "description": "Light quantization"},
            "q4_k_s": {"size_gb": 6.81, "vram_gb": 6, "description": "Recommended quantization", "recommended": True},
            "q4_0": {"size_gb": 6.79, "vram_gb": 6, "description": "Alternative Q4 quantization"},
            "q4_1": {"size_gb": 7.53, "vram_gb": 7, "description": "Higher quality Q4"},
            "q5_k_s": {"size_gb": 8.29, "vram_gb": 8, "description": "High quality quantization"},
            "q5_0": {"size_gb": 8.27, "vram_gb": 8, "description": "Alternative Q5 quantization"},
            "q5_1": {"size_gb": 9.01, "vram_gb": 9, "description": "Highest Q5 quality"},
            "q6_k": {"size_gb": 9.86, "vram_gb": 10, "description": "Very high quality"},
            "q8_0": {"size_gb": 12.7, "vram_gb": 12, "description": "Near-original quality"},
            "f16": {"size_gb": 23.8, "vram_gb": 24, "description": "Full precision"}
        }
        
        gguf_models = {}
        for variant, info in variants.items():
            model_name = f"flux.1-dev-gguf:{variant}"
            gguf_models[model_name] = {
                **base_gguf,
                "name": model_name,
                "variant": variant,
                "file_name": f"flux1-dev-{variant.upper()}.gguf",
                "quantization": variant.upper(),
                "size_gb": info["size_gb"],
                "description": f"{base_gguf['description']} - {info['description']}",
                "hardware_requirements": {
                    "min_vram_gb": info["vram_gb"],
                    "recommended_vram_gb": info["vram_gb"] + 2,
                    "min_ram_gb": 8,
                    "recommended_ram_gb": 16
                },
                "parameters": {
                    "num_inference_steps": 16,
                    "guidance_scale": 2.0,
                    "max_sequence_length": 512
                },
                "downloads": 50000 - (info["vram_gb"] * 1000),  # Simulate popularity
                "recommended": info.get("recommended", False)
            }
        
        return gguf_models
    
    def _load_cache(self):
        """Load cached model registry"""
        try:
            if self.registry_cache_file.exists():
                with open(self.registry_cache_file, 'r') as f:
                    self._cached_models = json.load(f)
                logger.debug(f"Loaded {len(self._cached_models)} models from cache")
        except Exception as e:
            logger.warning(f"Failed to load model cache: {e}")
            self._cached_models = {}
    
    def _save_cache(self, models: Dict[str, Any]):
        """Save model registry to cache"""
        try:
            with open(self.registry_cache_file, 'w') as f:
                json.dump(models, f, indent=2)
            
            with open(self.last_update_file, 'w') as f:
                f.write(datetime.now().isoformat())
                
            logger.debug(f"Saved {len(models)} models to cache")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")
    
    def _is_cache_expired(self) -> bool:
        """Check if cache is expired"""
        try:
            if not self.last_update_file.exists():
                return True
            
            with open(self.last_update_file, 'r') as f:
                last_update = datetime.fromisoformat(f.read().strip())
            
            return datetime.now() - last_update > self.cache_duration
        except:
            return True
    
    def _fetch_from_source(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch models from a specific source"""
        try:
            logger.debug(f"Fetching models from {source['name']}: {source['url']}")
            
            response = requests.get(
                source['url'], 
                timeout=source['timeout'],
                headers={'User-Agent': 'OllamaDiffuser/1.0'}
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Normalize the data format
            if 'models' in data:
                models = data['models']
            elif isinstance(data, dict):
                models = data
            else:
                logger.warning(f"Unexpected data format from {source['name']}")
                return None
            
            logger.info(f"Fetched {len(models)} models from {source['name']}")
            return models
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching from {source['name']}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch from {source['name']}: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from {source['name']}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error fetching from {source['name']}: {e}")
        
        return None
    
    def refresh(self, force: bool = False) -> bool:
        """Refresh model registry from external sources"""
        if not force and not self._is_cache_expired():
            logger.debug("Cache is still fresh, skipping refresh")
            return True
        
        logger.info("Refreshing model registry...")
        
        # Try each source in priority order
        for source in self.registry_sources:
            if not source.get('enabled', True):
                continue
            
            models = self._fetch_from_source(source)
            if models:
                # Merge with built-in models
                combined_models = {**self._builtin_models, **models}
                
                # Update cache
                self._cached_models = combined_models
                self._save_cache(combined_models)
                
                logger.info(f"Successfully updated registry with {len(combined_models)} models")
                return True
        
        logger.warning("Failed to fetch from any source, using cached/builtin models")
        return False
    
    def get_models(self, refresh: bool = False) -> Dict[str, Any]:
        """Get all available models"""
        if refresh or not self._cached_models:
            self.refresh()
        
        # Return cached models if available, otherwise built-in
        return self._cached_models if self._cached_models else self._builtin_models
    
    def get_model(self, model_name: str, refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get specific model information"""
        models = self.get_models(refresh=refresh)
        return models.get(model_name)
    
    def search_models(self, query: str = "", tags: List[str] = None, model_type: str = None) -> Dict[str, Any]:
        """Search models by query, tags, or type"""
        models = self.get_models()
        results = {}
        
        query_lower = query.lower()
        tags = tags or []
        
        for name, model in models.items():
            # Check query match
            query_match = (
                not query or 
                query_lower in name.lower() or 
                query_lower in model.get('description', '').lower()
            )
            
            # Check type match
            type_match = not model_type or model.get('model_type') == model_type
            
            # Check tags match
            model_tags = model.get('tags', [])
            tags_match = not tags or any(tag in model_tags for tag in tags)
            
            if query_match and type_match and tags_match:
                results[name] = model
        
        return results
    
    def get_popular_models(self, limit: int = 10) -> Dict[str, Any]:
        """Get most popular models"""
        models = self.get_models()
        
        # Sort by downloads
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1].get('downloads', 0),
            reverse=True
        )
        
        return dict(sorted_models[:limit])
    
    def get_recommended_models(self) -> Dict[str, Any]:
        """Get recommended models"""
        models = self.get_models()
        
        return {
            name: model for name, model in models.items()
            if model.get('recommended', False)
        }
    
    def add_local_model(self, model_name: str, model_config: Dict[str, Any]):
        """Add a local model configuration"""
        # Add to current models
        current_models = self.get_models()
        current_models[model_name] = model_config
        
        # Save to cache
        self._save_cache(current_models)
        self._cached_models = current_models
        
        logger.info(f"Added local model: {model_name}")
    
    def remove_local_model(self, model_name: str) -> bool:
        """Remove a local model configuration"""
        current_models = self.get_models()
        
        if model_name in current_models:
            del current_models[model_name]
            self._save_cache(current_models)
            self._cached_models = current_models
            logger.info(f"Removed local model: {model_name}")
            return True
        
        return False

# Global registry instance
model_registry = ModelRegistry() 