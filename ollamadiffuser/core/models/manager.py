import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import hashlib
from huggingface_hub import snapshot_download, login, hf_hub_download
from ..config.settings import settings, ModelConfig

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.loaded_model: Optional[object] = None
        self.current_model_name: Optional[str] = None
        
        # 预定义的模型库
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
        """列出所有可用的模型"""
        return list(self.model_registry.keys())
    
    def list_installed_models(self) -> List[str]:
        """列出已安装的模型"""
        return list(settings.models.keys())
    
    def is_model_installed(self, model_name: str) -> bool:
        """检查模型是否已安装"""
        return model_name in settings.models
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """获取模型信息"""
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
        """获取模型大小"""
        try:
            path = Path(model_path)
            if path.is_file():
                size = path.stat().st_size
            else:
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            # 转换为人类可读格式
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} PB"
        except Exception:
            return "Unknown"
    
    def pull_model(self, model_name: str, force: bool = False) -> bool:
        """下载模型"""
        if not force and self.is_model_installed(model_name):
            logger.info(f"模型 {model_name} 已存在")
            return True
        
        if model_name not in self.model_registry:
            logger.error(f"未知模型: {model_name}")
            return False
        
        model_info = self.model_registry[model_name]
        model_path = settings.get_model_path(model_name)
        
        try:
            # 确保HuggingFace token已设置
            if settings.hf_token:
                login(token=settings.hf_token)
            
            logger.info(f"正在下载模型: {model_name}")
            
            # 下载主模型
            snapshot_download(
                repo_id=model_info["repo_id"],
                local_dir=model_path,
                local_dir_use_symlinks=False,
                cache_dir=settings.cache_dir
            )
            
            # 下载组件 (如 LoRA)
            if "components" in model_info:
                components_path = model_path / "components"
                components_path.mkdir(exist_ok=True)
                
                for comp_name, comp_info in model_info["components"].items():
                    comp_path = components_path / comp_name
                    comp_path.mkdir(exist_ok=True)
                    
                    if "filename" in comp_info:
                        # 下载单个文件
                        hf_hub_download(
                            repo_id=comp_info["repo_id"],
                            filename=comp_info["filename"],
                            local_dir=comp_path,
                            cache_dir=settings.cache_dir
                        )
                    else:
                        # 下载整个仓库
                        snapshot_download(
                            repo_id=comp_info["repo_id"],
                            local_dir=comp_path,
                            local_dir_use_symlinks=False,
                            cache_dir=settings.cache_dir
                        )
            
            # 添加到配置
            model_config = ModelConfig(
                name=model_name,
                path=str(model_path),
                model_type=model_info["model_type"],
                variant=model_info.get("variant"),
                components=model_info.get("components"),
                parameters=model_info.get("parameters")
            )
            
            settings.add_model(model_config)
            logger.info(f"模型 {model_name} 下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            # 清理失败的下载
            if model_path.exists():
                shutil.rmtree(model_path)
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """删除模型"""
        if not self.is_model_installed(model_name):
            logger.error(f"模型 {model_name} 未安装")
            return False
        
        try:
            # 如果当前正在使用这个模型，先卸载
            if self.current_model_name == model_name:
                self.unload_model()
            
            # 删除模型文件
            model_config = settings.models[model_name]
            model_path = Path(model_config.path)
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # 从配置中移除
            settings.remove_model(model_name)
            
            logger.info(f"模型 {model_name} 已删除")
            return True
            
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def load_model(self, model_name: str) -> bool:
        """加载模型到内存"""
        if not self.is_model_installed(model_name):
            logger.error(f"模型 {model_name} 未安装")
            return False
        
        # 如果已经加载了相同的模型，直接返回
        if self.current_model_name == model_name:
            logger.info(f"模型 {model_name} 已加载")
            return True
        
        # 卸载当前模型
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
                logger.info(f"模型 {model_name} 加载成功")
                return True
            else:
                logger.error(f"模型 {model_name} 加载失败")
                return False
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def unload_model(self):
        """卸载当前模型"""
        if self.loaded_model is not None:
            try:
                self.loaded_model.unload()
                logger.info(f"模型 {self.current_model_name} 已卸载")
            except Exception as e:
                logger.error(f"卸载模型失败: {e}")
            finally:
                self.loaded_model = None
                self.current_model_name = None
    
    def get_current_model(self) -> Optional[str]:
        """获取当前加载的模型名称"""
        return self.current_model_name
    
    def is_model_loaded(self) -> bool:
        """检查是否有模型已加载"""
        return self.loaded_model is not None

# 全局模型管理器实例
model_manager = ModelManager() 