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
    """推理引擎，负责实际的图像生成"""
    
    def __init__(self):
        self.pipeline = None
        self.model_config: Optional[ModelConfig] = None
        self.device = None
        self.tokenizer = None
        self.max_token_limit = 77
        
    def _get_device(self) -> str:
        """自动检测可用设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    
    def _get_pipeline_class(self, model_type: str):
        """根据模型类型获取对应的pipeline类"""
        pipeline_map = {
            "sd15": StableDiffusionPipeline,
            "sdxl": StableDiffusionXLPipeline,
            "sd3": StableDiffusion3Pipeline
        }
        return pipeline_map.get(model_type)
    
    def load_model(self, model_config: ModelConfig) -> bool:
        """加载模型"""
        try:
            self.device = self._get_device()
            logger.info(f"使用设备: {self.device}")
            
            # 获取对应的pipeline类
            pipeline_class = self._get_pipeline_class(model_config.model_type)
            if not pipeline_class:
                logger.error(f"不支持的模型类型: {model_config.model_type}")
                return False
            
            # 设置加载参数
            load_kwargs = {}
            if model_config.variant == "fp16":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["variant"] = "fp16"
            
            # 加载pipeline
            logger.info(f"正在加载模型: {model_config.name}")
            self.pipeline = pipeline_class.from_pretrained(
                model_config.path,
                **load_kwargs
            ).to(self.device)
            
            # 加载LoRA等组件
            if model_config.components and "lora" in model_config.components:
                self._load_lora(model_config)
            
            # 优化设置
            self._apply_optimizations()
            
            # 设置tokenizer
            if hasattr(self.pipeline, 'tokenizer'):
                self.tokenizer = self.pipeline.tokenizer
            
            self.model_config = model_config
            logger.info(f"模型 {model_config.name} 加载完成")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def _load_lora(self, model_config: ModelConfig):
        """加载LoRA权重"""
        try:
            lora_config = model_config.components["lora"]
            components_path = Path(model_config.path) / "components" / "lora"
            
            if "filename" in lora_config:
                # 从本地文件加载
                lora_path = components_path / lora_config["filename"]
                if lora_path.exists():
                    self.pipeline.load_lora_weights(str(components_path), weight_name=lora_config["filename"])
                    self.pipeline.fuse_lora()
                    logger.info("LoRA权重加载完成")
            else:
                # 从目录加载
                if components_path.exists():
                    self.pipeline.load_lora_weights(str(components_path))
                    self.pipeline.fuse_lora()
                    logger.info("LoRA权重加载完成")
                    
        except Exception as e:
            logger.warning(f"加载LoRA权重失败: {e}")
    
    def _apply_optimizations(self):
        """应用性能优化"""
        try:
            # 启用注意力切片以节省内存
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            # 启用torch.compile加速（MPS不支持）
            if hasattr(torch, 'compile') and self.device != "mps":
                if hasattr(self.pipeline, 'unet'):
                    self.pipeline.unet = torch.compile(
                        self.pipeline.unet, 
                        mode="reduce-overhead", 
                        fullgraph=True
                    )
                    logger.info("torch.compile优化已启用")
                    
        except Exception as e:
            logger.warning(f"应用优化设置失败: {e}")
    
    def truncate_prompt(self, prompt: str) -> str:
        """截断prompt以适应CLIP token限制"""
        if not prompt or not self.tokenizer:
            return prompt
        
        # 对prompt进行编码
        tokens = self.tokenizer.encode(prompt)
        
        # 检查是否需要截断
        if len(tokens) <= self.max_token_limit:
            return prompt
        
        # 截断tokens并解码回文本
        truncated_tokens = tokens[:self.max_token_limit]
        truncated_prompt = self.tokenizer.decode(truncated_tokens)
        
        logger.warning(f"Prompt已截断: {len(tokens)} -> {len(truncated_tokens)} tokens")
        return truncated_prompt
    
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
                      num_inference_steps: Optional[int] = None,
                      guidance_scale: Optional[float] = None,
                      width: int = 1024,
                      height: int = 1024,
                      **kwargs) -> Image.Image:
        """生成图像"""
        if not self.pipeline:
            raise RuntimeError("模型未加载")
        
        # 使用模型默认参数
        if num_inference_steps is None:
            num_inference_steps = self.model_config.parameters.get("num_inference_steps", 28)
        
        if guidance_scale is None:
            guidance_scale = self.model_config.parameters.get("guidance_scale", 3.5)
        
        # 截断prompts
        truncated_prompt = self.truncate_prompt(prompt)
        truncated_negative_prompt = self.truncate_prompt(negative_prompt)
        
        try:
            logger.info(f"开始生成图像: {truncated_prompt[:50]}...")
            
            # 生成参数
            generation_kwargs = {
                "prompt": truncated_prompt,
                "negative_prompt": truncated_negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                **kwargs
            }
            
            # 为SDXL和SD3添加尺寸参数
            if self.model_config.model_type in ["sdxl", "sd3"]:
                generation_kwargs.update({
                    "width": width,
                    "height": height
                })
            
            # 生成图像
            output = self.pipeline(**generation_kwargs)
            
            image = output.images[0]
            logger.info("图像生成完成")
            return image
            
        except Exception as e:
            logger.error(f"图像生成失败: {e}")
            # 返回错误图像
            return self._create_error_image(str(e), truncated_prompt)
    
    def _create_error_image(self, error_msg: str, prompt: str) -> Image.Image:
        """创建错误提示图像"""
        from PIL import ImageDraw, ImageFont
        
        # 创建白色背景图像
        img = Image.new('RGB', (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 绘制错误信息
        try:
            # 尝试使用系统字体
            font = ImageFont.load_default()
        except:
            font = None
        
        # 绘制文本
        draw.text((10, 10), f"Error: {error_msg}", fill=(255, 0, 0), font=font)
        draw.text((10, 30), f"Prompt: {prompt[:50]}...", fill=(0, 0, 0), font=font)
        
        return img
    
    def unload(self):
        """卸载模型，释放显存"""
        if self.pipeline:
            # 移动到CPU以释放GPU内存
            self.pipeline = self.pipeline.to("cpu")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 删除pipeline
            del self.pipeline
            self.pipeline = None
            self.model_config = None
            self.tokenizer = None
            
            logger.info("模型已卸载")
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.pipeline is not None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前加载的模型信息"""
        if not self.model_config:
            return None
        
        return {
            "name": self.model_config.name,
            "type": self.model_config.model_type,
            "device": self.device,
            "variant": self.model_config.variant,
            "parameters": self.model_config.parameters
        } 