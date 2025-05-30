# ControlNet Guide for OllamaDiffuser

This comprehensive guide covers everything you need to know about using ControlNet with OllamaDiffuser for precise image generation control.

## 📋 Table of Contents

- [What is ControlNet?](#what-is-controlnet)
- [Installation](#installation)
- [Available ControlNet Models](#available-controlnet-models)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Control Types and Preprocessors](#control-types-and-preprocessors)
- [Parameters and Fine-tuning](#parameters-and-fine-tuning)
- [Web UI Usage](#web-ui-usage)
- [API Usage](#api-usage)
- [Performance and Optimization](#performance-and-optimization)
- [Troubleshooting](#troubleshooting)

## 🎛️ What is ControlNet?

ControlNet is a neural network architecture that allows you to control image generation using additional input images like edge maps, depth maps, pose keypoints, and more. This enables precise control over composition, structure, and style while maintaining the creative power of diffusion models.

### Key Benefits:
- **Precise Control**: Guide generation with specific structural elements
- **Consistent Composition**: Maintain layout and pose across generations
- **Creative Flexibility**: Combine multiple control types for complex scenes
- **Professional Results**: Achieve predictable, high-quality outputs

## 🚀 Installation

### Prerequisites
```bash
# Ensure you have OllamaDiffuser installed
pip install -e .

# ControlNet dependencies are included automatically
# controlnet-aux>=0.0.7 (advanced preprocessors)
# opencv-python>=4.8.0 (basic image processing)
# diffusers>=0.26.0 (ControlNet support)
```

### ⚡ Lazy Loading Design
**New in v1.1.0**: ControlNet preprocessors use lazy loading for optimal performance:

- **Fast Startup**: `ollamadiffuser --help` runs instantly without downloading models
- **On-Demand Loading**: Preprocessors only initialize when actually needed
- **User Control**: Explicit initialization available for pre-loading
- **Automatic Loading**: Seamless initialization when uploading control images

## 📦 Available ControlNet Models

### Stable Diffusion 1.5 ControlNet Models
```bash
# Edge-based control
ollamadiffuser pull controlnet-canny-sd15

# Depth-based control  
ollamadiffuser pull controlnet-depth-sd15

# Human pose control
ollamadiffuser pull controlnet-openpose-sd15

# Sketch-based control
ollamadiffuser pull controlnet-scribble-sd15
```

### Stable Diffusion XL ControlNet Models
```bash
# Edge-based control for SDXL
ollamadiffuser pull controlnet-canny-sdxl

# Depth-based control for SDXL
ollamadiffuser pull controlnet-depth-sdxl
```

### Model Requirements
Each ControlNet model requires its corresponding base model:
- **SD 1.5 ControlNet**: Requires `stable-diffusion-1.5`
- **SDXL ControlNet**: Requires `stable-diffusion-xl`

## 🏃 Quick Start

### 1. Install Base and ControlNet Models
```bash
# Install base model first
ollamadiffuser pull stable-diffusion-1.5

# Install ControlNet model
ollamadiffuser pull controlnet-canny-sd15
```

### 2. Load ControlNet Model
```bash
# Load the ControlNet model
ollamadiffuser load controlnet-canny-sd15
```

### 3. Generate with Control
```bash
# Start the server
ollamadiffuser run controlnet-canny-sd15

# Generate with control image (in another terminal)
curl -X POST http://localhost:8000/api/generate/controlnet \
  -F "prompt=a beautiful landscape, photorealistic" \
  -F "control_image=@your_control_image.jpg" \
  -F "controlnet_conditioning_scale=1.0"
```

## 💡 Usage Examples

### CLI Usage
```bash
# Load ControlNet model
ollamadiffuser load controlnet-canny-sd15

# Start server
ollamadiffuser run controlnet-canny-sd15

# Generate with control image
curl -X POST http://localhost:8000/api/generate/controlnet \
  -F "prompt=a futuristic city, cyberpunk style" \
  -F "negative_prompt=blurry, low quality" \
  -F "control_image=@edge_map.jpg" \
  -F "width=512" \
  -F "height=512" \
  -F "controlnet_conditioning_scale=1.2" \
  -F "num_inference_steps=20"
```

### Python API Usage
```python
from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor
from PIL import Image

# Load ControlNet model
model_manager.load_model("controlnet-canny-sd15")
engine = model_manager.loaded_model

# Load and preprocess control image
control_image = Image.open("input.jpg")

# Initialize ControlNet preprocessors (optional - auto-initializes when needed)
controlnet_preprocessor.initialize()

# Preprocess for canny edges
processed_image = controlnet_preprocessor.preprocess(control_image, "canny")

# Generate image
result = engine.generate_image(
    prompt="a beautiful garden, impressionist painting",
    control_image=processed_image,
    controlnet_conditioning_scale=1.0,
    width=512,
    height=512
)

result.save("output.jpg")
```

### Web UI Usage
1. **Start Web UI**: `ollamadiffuser --mode ui`
2. **Load ControlNet Model**: Select and load a ControlNet model
3. **Upload Control Image**: Use the file upload in the ControlNet section
4. **Adjust Parameters**: Set conditioning scale and guidance timing
5. **Generate**: Click "Generate Image" to create controlled output

#### ⚡ Lazy Loading in Web UI
- **Status Indicator**: Shows if ControlNet preprocessors are initialized
- **Auto-Initialization**: Preprocessors load automatically when you upload an image
- **Manual Initialization**: Click "Initialize ControlNet Preprocessors" for faster processing
- **Real-time Feedback**: Clear status messages and progress indicators

## 🎨 Control Types and Preprocessors

### Available Control Types

#### 1. **Canny Edge Detection**
- **Use Case**: Structural control, architectural elements
- **Best For**: Buildings, objects with clear edges
- **Parameters**: `low_threshold`, `high_threshold`

#### 2. **Depth Estimation**
- **Use Case**: 3D structure control, depth-based composition
- **Best For**: Landscapes, portraits, 3D scenes
- **Preprocessor**: MiDaS depth estimation

#### 3. **OpenPose Human Detection**
- **Use Case**: Human pose control, character positioning
- **Best For**: Portraits, figure drawing, character art
- **Detects**: Body keypoints, hand poses, facial landmarks

#### 4. **Scribble/Sketch Control**
- **Use Case**: Artistic sketches, rough compositions
- **Best For**: Creative workflows, concept art
- **Preprocessor**: HED edge detection

#### 5. **Advanced Control Types** (when preprocessors are initialized)
- **HED**: Holistically-nested edge detection
- **MLSD**: Mobile line segment detection
- **Normal**: Surface normal estimation
- **Lineart**: Clean line art detection
- **Lineart Anime**: Anime-style line art
- **Shuffle**: Content shuffling for style transfer

### Preprocessor Initialization

#### Automatic Initialization
```python
# Preprocessors initialize automatically when needed
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor

# Check availability without initializing
print("Available:", controlnet_preprocessor.is_available())
print("Initialized:", controlnet_preprocessor.is_initialized())

# Use preprocess() - auto-initializes if needed
processed = controlnet_preprocessor.preprocess(image, "canny")
```

#### Manual Initialization
```python
# Explicit initialization for faster subsequent processing
success = controlnet_preprocessor.initialize()
print("Initialization successful:", success)
print("Available types:", controlnet_preprocessor.get_available_types())
```

#### Fallback Processing
If advanced preprocessors fail to initialize, the system gracefully falls back to basic OpenCV-based processing for core control types (canny, depth, scribble).

## ⚙️ Parameters and Fine-tuning

### Core ControlNet Parameters

#### `controlnet_conditioning_scale` (0.0 - 2.0)
Controls how strongly the control image influences generation:
- **0.5-0.7**: Subtle control, more creative freedom
- **0.8-1.0**: Normal control (recommended)
- **1.1-1.5**: Strong control, strict adherence
- **1.6-2.0**: Very strong control, may reduce quality

#### `control_guidance_start` (0.0 - 1.0)
When to start applying control during generation:
- **0.0**: Apply control from the beginning (default)
- **0.2-0.3**: Allow initial creative freedom, then apply control
- **0.5+**: Apply control only in later stages

#### `control_guidance_end` (0.0 - 1.0)
When to stop applying control:
- **1.0**: Apply control until the end (default)
- **0.7-0.9**: Allow creative finishing touches
- **0.5**: Apply control only in early stages

### Example Parameter Combinations

#### Strict Architectural Control
```python
controlnet_conditioning_scale=1.3
control_guidance_start=0.0
control_guidance_end=1.0
```

#### Creative Portrait with Pose Guidance
```python
controlnet_conditioning_scale=0.8
control_guidance_start=0.0
control_guidance_end=0.8
```

#### Loose Artistic Interpretation
```python
controlnet_conditioning_scale=0.6
control_guidance_start=0.2
control_guidance_end=0.7
```

## 🌐 Web UI Usage

### Getting Started
1. **Launch Web UI**:
   ```bash
   ollamadiffuser --mode ui
   # Open http://localhost:8001 in your browser
   ```

2. **Load ControlNet Model**:
   - Select a ControlNet model from the dropdown
   - Click "🚀 Load Model"
   - Wait for the ControlNet indicator: 🎛️ ControlNet (type)

### ControlNet Section Features

#### Initialization Status
- **⚠️ Not Initialized**: Shows warning with initialization button
- **✅ Ready**: Preprocessors are loaded and ready
- **Auto-Initialize**: Happens automatically when uploading images

#### File Upload
- **Supported Formats**: JPG, PNG, WebP, BMP
- **Auto-Processing**: Images are automatically preprocessed based on ControlNet type
- **Preview**: Side-by-side display of control and generated images

#### Parameter Controls
- **Conditioning Scale**: Slider with real-time preview
- **Guidance Timing**: Start/end controls with helpful tooltips
- **Responsive Design**: Works on desktop and mobile devices

### Advanced Web UI Features

#### Real-time Status
- Model loading status
- ControlNet type detection
- Preprocessor initialization status
- Generation progress

#### Error Handling
- Clear error messages for common issues
- Automatic fallback for missing dependencies
- Validation for file uploads and parameters

## 🔌 API Usage

### REST API Endpoints

#### Initialize ControlNet Preprocessors
```bash
POST /api/controlnet/initialize
```
```json
{
  "success": true,
  "initialized": true,
  "available_types": ["canny", "depth", "openpose", ...],
  "message": "ControlNet preprocessors initialized successfully"
}
```

#### Get Available Preprocessors
```bash
GET /api/controlnet/preprocessors
```
```json
{
  "available_types": ["canny", "depth", "openpose", "hed", "mlsd", "normal", "lineart", "lineart_anime", "shuffle", "scribble"],
  "available": true,
  "initialized": true,
  "description": {
    "canny": "Edge detection using Canny algorithm",
    "depth": "Depth estimation for depth-based control",
    ...
  }
}
```

#### Preprocess Control Image
```bash
POST /api/controlnet/preprocess
Content-Type: multipart/form-data

control_type: "canny"
image: <file>
```
Returns preprocessed image as PNG.

#### Generate with ControlNet
```bash
POST /api/generate/controlnet
Content-Type: multipart/form-data

prompt: "a beautiful landscape"
control_image: <file>
controlnet_conditioning_scale: 1.0
control_guidance_start: 0.0
control_guidance_end: 1.0
width: 512
height: 512
num_inference_steps: 20
guidance_scale: 7.5
```

### Python API Integration
```python
import requests
from PIL import Image
import io

# Initialize ControlNet (optional)
response = requests.post("http://localhost:8000/api/controlnet/initialize")
print(response.json())

# Preprocess image
with open("control_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/controlnet/preprocess",
        data={"control_type": "canny"},
        files={"image": f}
    )

# Save preprocessed image
if response.status_code == 200:
    processed_image = Image.open(io.BytesIO(response.content))
    processed_image.save("preprocessed.png")

# Generate with ControlNet
with open("control_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/generate/controlnet",
        data={
            "prompt": "a futuristic cityscape",
            "controlnet_conditioning_scale": 1.0,
            "width": 512,
            "height": 512
        },
        files={"control_image": f}
    )

# Save generated image
if response.status_code == 200:
    result_image = Image.open(io.BytesIO(response.content))
    result_image.save("generated.png")
```

## 🚀 Performance and Optimization

### Lazy Loading Benefits
- **Instant Startup**: Application starts immediately without downloading models
- **Memory Efficient**: Only loads preprocessors when actually needed
- **User Choice**: Explicit control over when to initialize
- **Bandwidth Saving**: No unnecessary downloads for users who don't use ControlNet

### Initialization Strategies

#### For Interactive Use
```python
# Let preprocessors initialize automatically when needed
# No action required - happens seamlessly
```

#### For Batch Processing
```python
# Pre-initialize for faster processing
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor

# Initialize once at the start
controlnet_preprocessor.initialize()

# Process multiple images quickly
for image_path in image_list:
    processed = controlnet_preprocessor.preprocess(image, "canny")
    # ... generate with processed image
```

#### For Production Deployment
```bash
# Pre-warm the system by initializing ControlNet
curl -X POST http://localhost:8000/api/controlnet/initialize

# Then handle user requests normally
```

### Hardware Recommendations

#### GPU Requirements
- **Minimum**: 6GB VRAM (SD 1.5 ControlNet)
- **Recommended**: 12GB VRAM (SDXL ControlNet)
- **Optimal**: 16GB+ VRAM (multiple ControlNet types)

#### CPU Fallback
- ControlNet works on CPU but is significantly slower
- Basic preprocessors (canny, depth, scribble) work well on CPU
- Advanced preprocessors may require GPU for reasonable performance

### Memory Optimization
```python
# Enable memory optimizations
import torch

# Clear cache between generations
torch.cuda.empty_cache()

# Use attention slicing (enabled automatically)
# Use CPU offloading for large models (enabled automatically)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **Preprocessors Not Initializing**
```
Error: Failed to initialize ControlNet preprocessors
```

**Solutions**:
- Check internet connection (models download from HuggingFace)
- Verify `controlnet-aux` installation: `pip install controlnet-aux>=0.0.7`
- Try manual initialization: `controlnet_preprocessor.initialize(force=True)`
- Check available disk space (models require ~2GB)

#### 2. **Slow Startup (Legacy Behavior)**
If you experience slow startup, ensure you're using the lazy loading version:
```python
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor
print("Initialized at import:", controlnet_preprocessor.is_initialized())  # Should be False
```

#### 3. **Control Image Not Working**
**Check**:
- Image format (use JPG, PNG, WebP, BMP)
- Image size (will be automatically resized)
- ControlNet model is loaded (not just base model)
- Control image is provided for ControlNet models

#### 4. **Poor Control Quality**
**Adjust Parameters**:
- Increase `controlnet_conditioning_scale` (try 1.2-1.5)
- Use full guidance range: `control_guidance_start=0.0, control_guidance_end=1.0`
- Try different control types for your use case
- Ensure control image has clear, relevant features

#### 5. **Memory Issues**
**Solutions**:
- Use smaller image sizes (512x512 instead of 1024x1024)
- Enable CPU offloading (automatic)
- Close other applications
- Use basic preprocessors instead of advanced ones

#### 6. **API Errors**
```
HTTPException: Failed to initialize ControlNet preprocessors
```

**Solutions**:
- Check server logs for detailed error messages
- Verify model installation: `ollamadiffuser list`
- Restart server: `ollamadiffuser stop && ollamadiffuser run model-name`
- Check API endpoint status: `GET /api/controlnet/preprocessors`

### Debug Mode
```bash
# Enable verbose logging
ollamadiffuser -v run controlnet-canny-sd15

# Check preprocessor status
python -c "
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor
print('Available:', controlnet_preprocessor.is_available())
print('Initialized:', controlnet_preprocessor.is_initialized())
print('Types:', controlnet_preprocessor.get_available_types())
"
```

### Performance Monitoring
```python
import time
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor

# Time initialization
start = time.time()
success = controlnet_preprocessor.initialize()
init_time = time.time() - start
print(f"Initialization: {init_time:.2f}s, Success: {success}")

# Time preprocessing
start = time.time()
processed = controlnet_preprocessor.preprocess(image, "canny")
process_time = time.time() - start
print(f"Preprocessing: {process_time:.2f}s")
```

## 📚 Additional Resources

### Example Scripts
- `examples/controlnet_example.py`: Basic ControlNet usage
- `examples/controlnet_webui_example.py`: Web UI demonstration

### Model Information
- **Base Models**: Required for ControlNet functionality
- **Model Registry**: Automatic dependency management
- **Version Compatibility**: Diffusers >= 0.26.0 required

### Community Resources
- **HuggingFace Hub**: Browse available ControlNet models
- **ControlNet Paper**: Original research and methodology
- **Community Examples**: User-generated content and tutorials

---

## 🎉 Conclusion

ControlNet in OllamaDiffuser provides powerful, precise control over image generation with an optimized user experience. The lazy loading system ensures fast startup times while maintaining full functionality when needed.

**Key Takeaways**:
- ⚡ **Fast Startup**: Instant application launch with lazy loading
- 🎛️ **Precise Control**: Multiple control types for different use cases
- 🌐 **Multiple Interfaces**: CLI, Python API, Web UI, and REST API
- 🔧 **Flexible Parameters**: Fine-tune control strength and timing
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🚀 **Production Ready**: Robust error handling and fallback mechanisms

Start with the quick start guide and experiment with different control types to find what works best for your creative workflow! 