from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from PIL import Image

from ..core.models.manager import model_manager
from ..core.config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API request models
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: int = 1024
    height: int = 1024
    control_image_path: Optional[str] = None  # Path to control image file
    controlnet_conditioning_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0

class LoadModelRequest(BaseModel):
    model_name: str

class LoadLoRARequest(BaseModel):
    lora_name: str
    repo_id: str
    weight_name: Optional[str] = None
    scale: float = 1.0

def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="OllamaDiffuser API",
        description="Image generation model management and inference API",
        version="1.0.0"
    )
    
    # Add CORS middleware
    if settings.server.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "OllamaDiffuser API",
            "version": "1.0.0",
            "description": "Image generation model management and inference API",
            "status": "running",
            "endpoints": {
                "documentation": "/docs",
                "openapi_schema": "/openapi.json",
                "health_check": "/api/health",
                "models": "/api/models",
                "generate": "/api/generate"
            },
            "usage": {
                "web_ui": "Use 'ollamadiffuser --mode ui' to start the web interface",
                "cli": "Use 'ollamadiffuser --help' for command line options",
                "api_docs": "Visit /docs for interactive API documentation"
            }
        }
    
    # Model management endpoints
    @app.get("/api/models")
    async def list_models():
        """List all models"""
        return {
            "available": model_manager.list_available_models(),
            "installed": model_manager.list_installed_models(),
            "current": model_manager.get_current_model()
        }
    
    @app.get("/api/models/running")
    async def get_running_model():
        """Get currently running model"""
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
        """Get model detailed information"""
        info = model_manager.get_model_info(model_name)
        if info is None:
            raise HTTPException(status_code=404, detail="Model does not exist")
        return info
    
    @app.post("/api/models/pull")
    async def pull_model(model_name: str):
        """Download model"""
        if model_manager.pull_model(model_name):
            return {"message": f"Model {model_name} downloaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to download model {model_name}")
    
    @app.post("/api/models/load")
    async def load_model(request: LoadModelRequest):
        """Load model"""
        if model_manager.load_model(request.model_name):
            return {"message": f"Model {request.model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load model {request.model_name}")
    
    @app.post("/api/models/unload")
    async def unload_model():
        """Unload current model"""
        model_manager.unload_model()
        return {"message": "Model unloaded"}
    
    @app.delete("/api/models/{model_name}")
    async def remove_model(model_name: str):
        """Remove model"""
        if model_manager.remove_model(model_name):
            return {"message": f"Model {model_name} removed successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to remove model {model_name}")
    
    # LoRA management endpoints
    @app.post("/api/lora/load")
    async def load_lora(request: LoadLoRARequest):
        """Load LoRA weights into current model"""
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=400, detail="No model loaded, please load a model first")
        
        try:
            # Get current loaded inference engine
            engine = model_manager.loaded_model
            
            # Load LoRA weights
            success = engine.load_lora_runtime(
                repo_id=request.repo_id,
                weight_name=request.weight_name,
                scale=request.scale
            )
            
            if success:
                return {"message": f"LoRA {request.lora_name} loaded successfully with scale {request.scale}"}
            else:
                raise HTTPException(status_code=400, detail=f"Failed to load LoRA {request.lora_name}")
                
        except Exception as e:
            logger.error(f"LoRA loading failed: {e}")
            raise HTTPException(status_code=500, detail=f"LoRA loading failed: {str(e)}")
    
    @app.post("/api/lora/unload")
    async def unload_lora():
        """Unload current LoRA weights"""
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=400, detail="No model loaded")
        
        try:
            # Get current loaded inference engine
            engine = model_manager.loaded_model
            
            # Unload LoRA weights
            success = engine.unload_lora()
            
            if success:
                return {"message": "LoRA weights unloaded successfully"}
            else:
                raise HTTPException(status_code=400, detail="Failed to unload LoRA weights")
                
        except Exception as e:
            logger.error(f"LoRA unloading failed: {e}")
            raise HTTPException(status_code=500, detail=f"LoRA unloading failed: {str(e)}")
    
    @app.get("/api/lora/status")
    async def get_lora_status():
        """Get current LoRA status"""
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            return {"loaded": False, "message": "No model loaded"}
        
        try:
            # Get current loaded inference engine
            engine = model_manager.loaded_model
            
            # Check tracked LoRA state
            if hasattr(engine, 'current_lora') and engine.current_lora:
                lora_info = engine.current_lora.copy()
                return {
                    "loaded": True,
                    "info": lora_info,
                    "message": "LoRA loaded"
                }
            else:
                return {
                    "loaded": False,
                    "info": None,
                    "message": "No LoRA loaded"
                }
                
        except Exception as e:
            logger.error(f"Failed to get LoRA status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get LoRA status: {str(e)}")
    
    # Image generation endpoints
    @app.post("/api/generate")
    async def generate_image(request: GenerateRequest):
        """Generate image"""
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=400, detail="No model loaded, please load a model first")
        
        try:
            # Get current loaded inference engine
            engine = model_manager.loaded_model
            
            # Generate image
            image = engine.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                control_image=request.control_image_path,
                controlnet_conditioning_scale=request.controlnet_conditioning_scale,
                control_guidance_start=request.control_guidance_start,
                control_guidance_end=request.control_guidance_end
            )
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(content=img_byte_arr, media_type="image/png")
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
    
    @app.post("/api/generate/controlnet")
    async def generate_image_with_controlnet(
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality, low resolution"),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        width: int = Form(1024),
        height: int = Form(1024),
        controlnet_conditioning_scale: float = Form(1.0),
        control_guidance_start: float = Form(0.0),
        control_guidance_end: float = Form(1.0),
        control_image: Optional[UploadFile] = File(None)
    ):
        """Generate image with ControlNet using uploaded control image"""
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=400, detail="No model loaded, please load a model first")
        
        try:
            # Get current loaded inference engine
            engine = model_manager.loaded_model
            
            # Check if this is a ControlNet model
            if not engine.is_controlnet_pipeline:
                raise HTTPException(status_code=400, detail="Current model is not a ControlNet model")
            
            # Process control image if provided
            control_image_pil = None
            if control_image:
                # Read uploaded image
                image_data = await control_image.read()
                control_image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Generate image
            image = engine.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                control_image=control_image_pil,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end
            )
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(content=img_byte_arr, media_type="image/png")
            
        except Exception as e:
            logger.error(f"ControlNet image generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"ControlNet image generation failed: {str(e)}")
    
    @app.post("/api/controlnet/initialize")
    async def initialize_controlnet():
        """Initialize ControlNet preprocessors"""
        try:
            from ..core.utils.controlnet_preprocessors import controlnet_preprocessor
            
            logger.info("Explicitly initializing ControlNet preprocessors...")
            success = controlnet_preprocessor.initialize(force=True)
            
            return {
                "success": success,
                "initialized": controlnet_preprocessor.is_initialized(),
                "available_types": controlnet_preprocessor.get_available_types(),
                "message": "ControlNet preprocessors initialized successfully" if success else "Failed to initialize ControlNet preprocessors"
            }
        except Exception as e:
            logger.error(f"Failed to initialize ControlNet preprocessors: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize ControlNet preprocessors: {str(e)}")
    
    @app.get("/api/controlnet/preprocessors")
    async def get_controlnet_preprocessors():
        """Get available ControlNet preprocessors"""
        try:
            from ..core.utils.controlnet_preprocessors import controlnet_preprocessor
            return {
                "available_types": controlnet_preprocessor.get_available_types(),
                "available": controlnet_preprocessor.is_available(),
                "initialized": controlnet_preprocessor.is_initialized(),
                "description": {
                    "canny": "Edge detection using Canny algorithm",
                    "depth": "Depth estimation for depth-based control",
                    "openpose": "Human pose detection for pose control",
                    "scribble": "Scribble/sketch detection for artistic control",
                    "hed": "Holistically-nested edge detection",
                    "mlsd": "Mobile line segment detection",
                    "normal": "Surface normal estimation",
                    "lineart": "Line art detection",
                    "lineart_anime": "Anime-style line art detection",
                    "shuffle": "Content shuffling for style transfer"
                }
            }
        except Exception as e:
            logger.error(f"Failed to get ControlNet preprocessors: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get ControlNet preprocessors: {str(e)}")
    
    @app.post("/api/controlnet/preprocess")
    async def preprocess_control_image(
        control_type: str = Form(...),
        image: UploadFile = File(...)
    ):
        """Preprocess image for ControlNet"""
        try:
            from ..core.utils.controlnet_preprocessors import controlnet_preprocessor
            
            # Initialize ControlNet preprocessors if needed
            if not controlnet_preprocessor.is_initialized():
                logger.info("Initializing ControlNet preprocessors for image preprocessing...")
                if not controlnet_preprocessor.initialize():
                    raise HTTPException(status_code=500, detail="Failed to initialize ControlNet preprocessors")
            
            # Read uploaded image
            image_data = await image.read()
            input_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Preprocess image
            processed_image = controlnet_preprocessor.preprocess(input_image, control_type)
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(content=img_byte_arr, media_type="image/png")
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {str(e)}")
    
    # Health check endpoints
    @app.get("/api/health")
    async def health_check():
        """Health check"""
        return {
            "status": "healthy",
            "model_loaded": model_manager.is_model_loaded(),
            "current_model": model_manager.get_current_model()
        }
    
    # Server management endpoints
    @app.post("/api/shutdown")
    async def shutdown_server():
        """Gracefully shutdown the server"""
        import os
        import signal
        import asyncio
        
        # Unload model first
        model_manager.unload_model()
        
        # Schedule server shutdown
        def shutdown():
            os.kill(os.getpid(), signal.SIGTERM)
        
        # Delay shutdown to allow response to be sent
        asyncio.get_event_loop().call_later(0.5, shutdown)
        
        return {"message": "Server shutting down..."}
    
    return app

def run_server(host: str = None, port: int = None):
    """Start server"""
    # Use default values from configuration
    host = host or settings.server.host
    port = port or settings.server.port
    
    # Create FastAPI application
    app = create_app()
    
    # Run server with uvicorn
    logger.info(f"Starting server: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_server() 