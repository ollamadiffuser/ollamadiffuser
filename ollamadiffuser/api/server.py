import litserve as ls
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import io
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..core.models.manager import model_manager
from ..core.config.settings import settings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API请求模型
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: int = 1024
    height: int = 1024

class LoadModelRequest(BaseModel):
    model_name: str

class ImageGenerationAPI(ls.LitAPI):
    """图像生成API"""
    
    def setup(self, device):
        """初始化API"""
        self.device = device
        logger.info("ImageGenerationAPI初始化完成")
    
    def decode_request(self, request):
        """解析请求"""
        return request
    
    def predict(self, params: GenerateRequest):
        """执行图像生成预测"""
        # 检查是否有模型加载
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=400, detail="没有模型已加载，请先加载模型")
        
        # 获取当前加载的推理引擎
        engine = model_manager.loaded_model
        
        # 生成图像
        image = engine.generate_image(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height
        )
        
        return image
    
    def encode_response(self, image):
        """编码响应"""
        # 将PIL图像转换为字节
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="OllamaDiffuser API",
        description="图像生成模型管理和推理API",
        version="1.0.0"
    )
    
    # 添加CORS中间件
    if settings.server.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # 模型管理端点
    @app.get("/api/models")
    async def list_models():
        """列出所有模型"""
        return {
            "available": model_manager.list_available_models(),
            "installed": model_manager.list_installed_models(),
            "current": model_manager.get_current_model()
        }
    
    @app.get("/api/models/running")
    async def get_running_model():
        """获取当前运行的模型"""
        if model_manager.is_model_loaded():
            engine = model_manager.loaded_model
            return {
                "model": model_manager.get_current_model(),
                "info": engine.get_model_info(),
                "loaded": True
            }
        else:
            return {"loaded": False}
    
    @app.get("/api/models/{model_name}")
    async def get_model_info(model_name: str):
        """获取模型详细信息"""
        info = model_manager.get_model_info(model_name)
        if info is None:
            raise HTTPException(status_code=404, detail="模型不存在")
        return info
    
    @app.post("/api/models/pull")
    async def pull_model(model_name: str):
        """下载模型"""
        if model_manager.pull_model(model_name):
            return {"message": f"模型 {model_name} 下载成功"}
        else:
            raise HTTPException(status_code=400, detail=f"下载模型 {model_name} 失败")
    
    @app.post("/api/models/load")
    async def load_model(request: LoadModelRequest):
        """加载模型"""
        if model_manager.load_model(request.model_name):
            return {"message": f"模型 {request.model_name} 加载成功"}
        else:
            raise HTTPException(status_code=400, detail=f"加载模型 {request.model_name} 失败")
    
    @app.post("/api/models/unload")
    async def unload_model():
        """卸载当前模型"""
        model_manager.unload_model()
        return {"message": "模型已卸载"}
    
    @app.delete("/api/models/{model_name}")
    async def remove_model(model_name: str):
        """删除模型"""
        if model_manager.remove_model(model_name):
            return {"message": f"模型 {model_name} 删除成功"}
        else:
            raise HTTPException(status_code=400, detail=f"删除模型 {model_name} 失败")
    
    # 图像生成端点
    @app.post("/api/generate")
    async def generate_image(request: GenerateRequest):
        """生成图像"""
        # 检查是否有模型加载
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=400, detail="没有模型已加载，请先加载模型")
        
        try:
            # 获取当前加载的推理引擎
            engine = model_manager.loaded_model
            
            # 生成图像
            image = engine.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height
            )
            
            # 将PIL图像转换为字节
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(content=img_byte_arr, media_type="image/png")
            
        except Exception as e:
            logger.error(f"图像生成失败: {e}")
            raise HTTPException(status_code=500, detail=f"图像生成失败: {str(e)}")
    
    # 健康检查端点
    @app.get("/api/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "model_loaded": model_manager.is_model_loaded(),
            "current_model": model_manager.get_current_model()
        }
    
    return app

def run_server(host: str = None, port: int = None):
    """启动服务器"""
    # 使用配置中的默认值
    host = host or settings.server.host
    port = port or settings.server.port
    
    # 创建LitServe API
    api = ImageGenerationAPI()
    server = ls.LitServer(api, accelerator="auto")
    
    # 注册FastAPI应用
    fastapi_app = create_app()
    
    # 运行服务器
    logger.info(f"启动服务器: http://{host}:{port}")
    server.run(host=host, port=port, generate_client_file=False)

if __name__ == "__main__":
    run_server() 