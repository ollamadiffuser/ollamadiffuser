from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import base64
from pathlib import Path

from ..core.models.manager import model_manager

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
        
        # Don't auto-load model on startup - let user choose
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_loaded
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
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
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
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
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
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_manager.is_model_loaded(),
            "success_message": success_message,
            "error_message": error_message if 'error_message' in locals() else None
        })
    
    return app 