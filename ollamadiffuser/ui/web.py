from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import base64
import logging
import json
from pathlib import Path
from PIL import Image

from ..core.models.manager import model_manager
from ..core.utils.lora_manager import lora_manager
from ..core.utils.controlnet_preprocessors import controlnet_preprocessor

logger = logging.getLogger(__name__)

# Get templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

def create_ui_app() -> FastAPI:
    """Create Web UI application"""
    app = FastAPI(title="OllamaDiffuser Web UI")
    
    # Mount static files for samples
    samples_dir = Path(__file__).parent / "samples"
    logger.info(f"Samples directory: {samples_dir}")
    logger.info(f"Samples directory exists: {samples_dir.exists()}")
    if samples_dir.exists():
        logger.info(f"Mounting samples directory: {samples_dir}")
        app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")
    else:
        logger.warning(f"Samples directory not found: {samples_dir}")
    
    def get_template_context(request: Request):
        """Get common template context"""
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        model_loaded = model_manager.is_model_loaded()
        
        # Get LoRA information
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        # Check if current model is ControlNet
        is_controlnet_model = False
        controlnet_type = None
        model_parameters = {}
        if current_model and model_loaded:
            engine = model_manager.loaded_model
            if hasattr(engine, 'is_controlnet_pipeline'):
                is_controlnet_model = engine.is_controlnet_pipeline
                if is_controlnet_model:
                    # Get ControlNet type from model info
                    model_info = model_manager.get_model_info(current_model)
                    controlnet_type = model_info.get('controlnet_type', 'canny') if model_info else 'canny'
            
            # Get model parameters for current model
            model_info = model_manager.get_model_info(current_model)
            if model_info and 'parameters' in model_info:
                model_parameters = model_info['parameters']
        
        # Get available ControlNet preprocessors (without initializing)
        available_preprocessors = controlnet_preprocessor.get_available_types()
        
        # Load sample metadata
        sample_metadata = {}
        metadata_file = samples_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    sample_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sample metadata: {e}")
        
        return {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_loaded,
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "is_controlnet_model": is_controlnet_model,
            "controlnet_type": controlnet_type,
            "available_preprocessors": available_preprocessors,
            "controlnet_available": controlnet_preprocessor.is_available(),
            "controlnet_initialized": controlnet_preprocessor.is_initialized(),
            "sample_metadata": sample_metadata,
            "model_parameters": model_parameters
        }
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Home page"""
        return templates.TemplateResponse("index.html", get_template_context(request))
    
    @app.post("/generate")
    async def generate_image_ui(
        request: Request,
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality, low resolution"),
        num_inference_steps: int = Form(28),
        guidance_scale: float = Form(3.5),
        width: int = Form(1024),
        height: int = Form(1024),
        control_image: UploadFile = File(None),
        controlnet_conditioning_scale: float = Form(1.0),
        control_guidance_start: float = Form(0.0),
        control_guidance_end: float = Form(1.0)
    ):
        """Generate image (Web UI)"""
        error_message = None
        image_b64 = None
        control_image_b64 = None
        
        try:
            # Check if model is actually loaded in memory
            if not model_manager.is_model_loaded():
                error_message = "No model loaded. Please load a model first using the model management section above."
            
            if not error_message:
                # Get inference engine
                engine = model_manager.loaded_model
                
                if engine is None:
                    error_message = "Model engine is not available. Please reload the model."
                else:
                    # Process control image if provided
                    control_image_pil = None
                    if control_image and control_image.filename:
                        # Initialize ControlNet preprocessors if needed
                        if not controlnet_preprocessor.is_initialized():
                            logger.info("Initializing ControlNet preprocessors for image processing...")
                            if not controlnet_preprocessor.initialize():
                                error_message = "Failed to initialize ControlNet preprocessors. Please check your installation."
                        
                        if not error_message:
                            # Read uploaded image
                            image_data = await control_image.read()
                            control_image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
                            
                            # Convert control image to base64 for display
                            img_buffer = io.BytesIO()
                            control_image_pil.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            control_image_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
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
                    
                    # Convert to base64
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    image_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                
        except Exception as e:
            error_message = f"Image generation failed: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "image_b64": image_b64,
            "control_image_b64": control_image_b64,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/preprocess_control_image")
    async def preprocess_control_image_ui(
        request: Request,
        control_type: str = Form(...),
        image: UploadFile = File(...)
    ):
        """Preprocess control image (Web UI)"""
        try:
            # Initialize ControlNet preprocessors if needed
            if not controlnet_preprocessor.is_initialized():
                logger.info("Initializing ControlNet preprocessors for image preprocessing...")
                if not controlnet_preprocessor.initialize():
                    return {"error": "Failed to initialize ControlNet preprocessors. Please check your installation."}
            
            # Read uploaded image
            image_data = await image.read()
            input_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Preprocess image
            processed_image = controlnet_preprocessor.preprocess(input_image, control_type)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            processed_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return StreamingResponse(io.BytesIO(img_buffer.getvalue()), media_type="image/png")
            
        except Exception as e:
            # Return error as JSON
            return {"error": f"Image preprocessing failed: {str(e)}"}
    
    @app.post("/load_model")
    async def load_model_ui(request: Request, model_name: str = Form(...)):
        """Load model (Web UI)"""
        success = False
        error_message = None
        
        try:
            if model_manager.load_model(model_name):
                success = True
            else:
                error_message = f"Failed to load model {model_name}"
        except Exception as e:
            error_message = f"Error loading model: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": f"Model {model_name} loaded successfully!" if success else None,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/unload_model")
    async def unload_model_ui(request: Request):
        """Unload current model (Web UI)"""
        try:
            current_model = model_manager.get_current_model()
            model_manager.unload_model()
            success_message = f"Model {current_model} unloaded successfully!" if current_model else "Model unloaded!"
            error_message = None
        except Exception as e:
            success_message = None
            error_message = f"Error unloading model: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": success_message,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/load_lora")
    async def load_lora_ui(request: Request, lora_name: str = Form(...), scale: float = Form(1.0)):
        """Load LoRA (Web UI)"""
        success = False
        error_message = None
        
        try:
            if lora_manager.load_lora(lora_name, scale=scale):
                success = True
            else:
                error_message = f"Failed to load LoRA {lora_name}"
        except Exception as e:
            error_message = f"Error loading LoRA: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": f"LoRA {lora_name} loaded successfully with scale {scale}!" if success else None,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/unload_lora")
    async def unload_lora_ui(request: Request):
        """Unload current LoRA (Web UI)"""
        try:
            current_lora_name = lora_manager.get_current_lora()
            lora_manager.unload_lora()
            success_message = f"LoRA {current_lora_name} unloaded successfully!" if current_lora_name else "LoRA unloaded!"
            error_message = None
        except Exception as e:
            success_message = None
            error_message = f"Error unloading LoRA: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": success_message,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/pull_lora")
    async def pull_lora_ui(request: Request, repo_id: str = Form(...), weight_name: str = Form(""), alias: str = Form("")):
        """Pull LoRA from Hugging Face Hub (Web UI)"""
        success = False
        error_message = None
        
        try:
            # Use alias if provided, otherwise use repo_id
            lora_alias = alias if alias.strip() else None
            weight_file = weight_name if weight_name.strip() else None
            
            if lora_manager.pull_lora(repo_id, weight_name=weight_file, alias=lora_alias):
                success = True
                final_name = lora_alias if lora_alias else repo_id.replace('/', '_')
            else:
                error_message = f"Failed to download LoRA {repo_id}"
        except Exception as e:
            error_message = f"Error downloading LoRA: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": f"LoRA {final_name if success else repo_id} downloaded successfully!" if success else None,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)

    @app.post("/api/controlnet/initialize")
    async def initialize_controlnet_api():
        """Initialize ControlNet preprocessors (API endpoint)"""
        try:
            success = controlnet_preprocessor.initialize()
            return {
                "success": success,
                "initialized": controlnet_preprocessor.is_initialized(),
                "message": "ControlNet preprocessors initialized successfully!" if success else "Failed to initialize ControlNet preprocessors"
            }
        except Exception as e:
            logger.error(f"Error initializing ControlNet: {e}")
            return {
                "success": False,
                "initialized": False,
                "message": f"Error initializing ControlNet: {str(e)}"
            }

    return app 