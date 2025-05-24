import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """单个模型的配置"""
    name: str
    path: str
    model_type: str  # "sd15", "sdxl", "sd3", "flux", etc.
    variant: Optional[str] = None  # "fp16", "fp32", etc.
    components: Optional[Dict[str, str]] = None  # LoRA, VAE, etc.
    parameters: Optional[Dict[str, Any]] = None  # default generation parameters

@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "localhost"
    port: int = 8000
    max_queue_size: int = 100
    timeout: int = 600
    enable_cors: bool = True

class Settings:
    """应用全局设置"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".ollamadiffuser"
        self.models_dir = self.config_dir / "models"
        self.cache_dir = self.config_dir / "cache"
        self.config_file = self.config_dir / "config.json"
        
        # 确保目录存在
        self.config_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 默认配置
        self.server = ServerConfig()
        self.models: Dict[str, ModelConfig] = {}
        self.current_model: Optional[str] = None
        self.hf_token: Optional[str] = os.environ.get('HF_TOKEN')
        
        # 加载配置文件
        self.load_config()
    
    def load_config(self):
        """从配置文件加载设置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 加载服务器配置
                if 'server' in config_data:
                    server_data = config_data['server']
                    self.server = ServerConfig(**server_data)
                
                # 加载模型配置
                if 'models' in config_data:
                    self.models = {
                        name: ModelConfig(**model_data)
                        for name, model_data in config_data['models'].items()
                    }
                
                self.current_model = config_data.get('current_model')
                
                logger.info(f"已加载配置文件: {self.config_file}")
                
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
    
    def save_config(self):
        """保存设置到配置文件"""
        try:
            config_data = {
                'server': {
                    'host': self.server.host,
                    'port': self.server.port,
                    'max_queue_size': self.server.max_queue_size,
                    'timeout': self.server.timeout,
                    'enable_cors': self.server.enable_cors
                },
                'models': {
                    name: {
                        'name': model.name,
                        'path': model.path,
                        'model_type': model.model_type,
                        'variant': model.variant,
                        'components': model.components,
                        'parameters': model.parameters
                    }
                    for name, model in self.models.items()
                },
                'current_model': self.current_model
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {self.config_file}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def add_model(self, model_config: ModelConfig):
        """添加模型配置"""
        self.models[model_config.name] = model_config
        self.save_config()
    
    def remove_model(self, model_name: str):
        """移除模型配置"""
        if model_name in self.models:
            del self.models[model_name]
            if self.current_model == model_name:
                self.current_model = None
            self.save_config()
    
    def set_current_model(self, model_name: str):
        """设置当前使用的模型"""
        if model_name in self.models:
            self.current_model = model_name
            self.save_config()
        else:
            raise ValueError(f"模型 '{model_name}' 不存在")
    
    def get_model_path(self, model_name: str) -> Path:
        """获取模型的存储路径"""
        return self.models_dir / model_name

# 全局设置实例
settings = Settings() 