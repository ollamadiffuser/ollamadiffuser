from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import base64
from pathlib import Path

from ..core.models.manager import model_manager
from ..core.utils.lora_manager import lora_manager

# Get templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

def create_ui_app() -> FastAPI:
    """Create Web UI application"""
    app = FastAPI(title="OllamaDiffuser Web UI")
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Home page"""
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        model_loaded = model_manager.is_model_loaded()
        
        # Get LoRA information
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        # Don't auto-load model on startup - let user choose
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_loaded,
            "installed_loras": installed_loras,
            "current_lora": current_lora
        })
    
    @app.post("/generate")
    async def generate_image_ui(
        request: Request,
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality, low resolution"),
        num_inference_steps: int = Form(28),
        guidance_scale: float = Form(3.5),
        width: int = Form(1024),
        height: int = Form(1024)
    ):
        """Generate image (Web UI)"""
        error_message = None
        image_b64 = None
        
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
                    # Generate image
                    image = engine.generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height
                    )
                    
                    # Convert to base64
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    image_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                
        except Exception as e:
            error_message = f"Image generation failed: {str(e)}"
        
        # Return result page
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "image_b64": image_b64,
            "error_message": error_message
        })
    
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
        
        # Redirect back to home page
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "success_message": f"Model {model_name} loaded successfully!" if success else None,
            "error_message": error_message
        })
    
    @app.post("/unload_model")
    async def unload_model_ui(request: Request):
        """Unload current model (Web UI)"""
        try:
            current_model = model_manager.get_current_model()
            model_manager.unload_model()
            success_message = f"Model {current_model} unloaded successfully!" if current_model else "Model unloaded!"
        except Exception as e:
            success_message = None
            error_message = f"Error unloading model: {str(e)}"
        
        # Redirect back to home page
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "success_message": success_message,
            "error_message": error_message if 'error_message' in locals() else None
        })
    
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
        
        # Redirect back to home page
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "success_message": f"LoRA {lora_name} loaded successfully with scale {scale}!" if success else None,
            "error_message": error_message
        })
    
    @app.post("/unload_lora")
    async def unload_lora_ui(request: Request):
        """Unload current LoRA (Web UI)"""
        try:
            current_lora_name = lora_manager.get_current_lora()
            lora_manager.unload_lora()
            success_message = f"LoRA {current_lora_name} unloaded successfully!" if current_lora_name else "LoRA unloaded!"
        except Exception as e:
            success_message = None
            error_message = f"Error unloading LoRA: {str(e)}"
        
        # Redirect back to home page
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "success_message": success_message,
            "error_message": error_message if 'error_message' in locals() else None
        })
    
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
        
        # Redirect back to home page
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "success_message": f"LoRA {final_name if success else repo_id} downloaded successfully!" if success else None,
            "error_message": error_message
        })

    return app 