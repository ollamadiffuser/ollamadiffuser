# OllamaDiffuser 🎨

A powerful, user-friendly image generation tool that brings together multiple Stable Diffusion models with seamless model management, LoRA support, and advanced ControlNet capabilities.

## ✨ Features

- **🚀 Fast Startup**: Instant application launch with lazy loading architecture
- **🎛️ ControlNet Support**: Precise image generation control with 10+ control types
- **🔄 LoRA Integration**: Dynamic LoRA loading and management
- **🌐 Multiple Interfaces**: CLI, Python API, Web UI, and REST API
- **📦 Model Management**: Easy installation and switching between models
- **⚡ Performance Optimized**: Memory-efficient with GPU acceleration
- **🎨 Professional Results**: High-quality image generation with fine-tuned control

## 🏃 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ollamadiffuser.git
cd ollamadiffuser

# Install dependencies
pip install -e .
```

### Basic Usage
```bash
# Install a model
ollamadiffuser pull stable-diffusion-1.5

# Load the model
ollamadiffuser load stable-diffusion-1.5

# Generate an image
ollamadiffuser generate "a beautiful sunset over mountains"

# Start web interface
ollamadiffuser --mode ui
```

### ControlNet Quick Start
```bash
# Install ControlNet model
ollamadiffuser pull controlnet-canny-sd15

# Load ControlNet model
ollamadiffuser load controlnet-canny-sd15

# Generate with control image
curl -X POST http://localhost:8000/api/generate/controlnet \
  -F "prompt=a beautiful landscape" \
  -F "control_image=@your_image.jpg"
```

## 🎛️ ControlNet Features

### ⚡ Lazy Loading Architecture
**New in v1.1.0**: ControlNet preprocessors use intelligent lazy loading:

- **Instant Startup**: `ollamadiffuser --help` runs immediately without downloading models
- **On-Demand Loading**: Preprocessors initialize only when actually needed
- **Automatic Initialization**: Seamless loading when uploading control images
- **User Control**: Manual initialization available for pre-loading

### Available Control Types
- **Canny Edge Detection**: Structural control with edge maps
- **Depth Estimation**: 3D structure control with depth maps
- **OpenPose**: Human pose and body position control
- **Scribble/Sketch**: Artistic control with hand-drawn inputs
- **Advanced Types**: HED, MLSD, Normal, Lineart, Anime Lineart, Content Shuffle

### ControlNet Models
```bash
# SD 1.5 ControlNet Models
ollamadiffuser pull controlnet-canny-sd15
ollamadiffuser pull controlnet-depth-sd15
ollamadiffuser pull controlnet-openpose-sd15
ollamadiffuser pull controlnet-scribble-sd15

# SDXL ControlNet Models
ollamadiffuser pull controlnet-canny-sdxl
ollamadiffuser pull controlnet-depth-sdxl
```

## 🔄 LoRA Support

### Dynamic LoRA Management
```bash
# Download LoRA from Hugging Face
ollamadiffuser lora pull "openfree/flux-chatgpt-ghibli-lora"

# Load LoRA with custom strength
ollamadiffuser lora load ghibli --scale 1.2

# Unload LoRA
ollamadiffuser lora unload
```

### Web UI LoRA Integration
- **Easy Download**: Enter Hugging Face repository ID
- **Strength Control**: Adjust LoRA influence with sliders
- **Real-time Loading**: Load/unload LoRAs without restarting
- **Alias Support**: Create custom names for your LoRAs

## 🌐 Multiple Interfaces

### Command Line Interface
```bash
# Generate with advanced parameters
ollamadiffuser generate \
  "a futuristic cityscape" \
  --negative-prompt "blurry, low quality" \
  --steps 30 \
  --guidance 7.5 \
  --width 1024 \
  --height 1024
```

### Web UI
```bash
# Start web interface
ollamadiffuser --mode ui
# Open http://localhost:8001
```

Features:
- **Responsive Design**: Works on desktop and mobile
- **Real-time Status**: Model and LoRA loading indicators
- **ControlNet Integration**: File upload with preprocessing
- **Parameter Controls**: Intuitive sliders and inputs

### REST API
```bash
# Start API server
ollamadiffuser --mode api

# Generate image
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape", "width": 1024, "height": 1024}'
```

### Python API
```python
from ollamadiffuser.core.models.manager import model_manager

# Load model
model_manager.load_model("stable-diffusion-1.5")
engine = model_manager.loaded_model

# Generate image
image = engine.generate_image(
    prompt="a beautiful sunset",
    width=1024,
    height=1024
)
image.save("output.jpg")
```

## 📦 Supported Models

### Base Models
- **Stable Diffusion 1.5**: Classic, reliable, fast
- **Stable Diffusion XL**: High-resolution, detailed
- **Stable Diffusion 3**: Latest architecture
- **FLUX.1**: State-of-the-art quality

### ControlNet Models
- **SD 1.5 ControlNet**: 4 control types (canny, depth, openpose, scribble)
- **SDXL ControlNet**: 2 control types (canny, depth)

### LoRA Support
- **Hugging Face Integration**: Direct download from HF Hub
- **Local LoRA Files**: Support for local .safetensors files
- **Dynamic Loading**: Load/unload without model restart
- **Strength Control**: Adjustable influence (0.1-2.0)

## ⚙️ Configuration

### Model Configuration
Models are automatically configured with optimal settings:
- **Memory Optimization**: Attention slicing, CPU offloading
- **Device Detection**: Automatic CUDA/MPS/CPU selection
- **Precision Handling**: FP16/BF16 support for efficiency
- **Safety Features**: NSFW filter bypass for creative freedom

### Performance Tuning
```bash
# Enable verbose logging
ollamadiffuser -v generate "test prompt"

# Check system status
ollamadiffuser status

# Monitor memory usage
ollamadiffuser info
```

## 🔧 Advanced Usage

### ControlNet Parameters
```python
# Fine-tune ControlNet behavior
image = engine.generate_image(
    prompt="architectural masterpiece",
    control_image=control_img,
    controlnet_conditioning_scale=1.2,  # Strength (0.0-2.0)
    control_guidance_start=0.0,         # When to start (0.0-1.0)
    control_guidance_end=1.0            # When to end (0.0-1.0)
)
```

### Batch Processing
```python
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor

# Pre-initialize for faster processing
controlnet_preprocessor.initialize()

# Process multiple images
for image_path in image_list:
    control_img = controlnet_preprocessor.preprocess(image, "canny")
    result = engine.generate_image(prompt, control_image=control_img)
    result.save(f"output_{i}.jpg")
```

### API Integration
```python
import requests

# Initialize ControlNet preprocessors
response = requests.post("http://localhost:8000/api/controlnet/initialize")

# Check available preprocessors
response = requests.get("http://localhost:8000/api/controlnet/preprocessors")
print(response.json()["available_types"])

# Generate with file upload
with open("control.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/generate/controlnet",
        data={"prompt": "beautiful landscape"},
        files={"control_image": f}
    )
```

## 📚 Model-Specific Guides

- **[ControlNet Guide](CONTROLNET_GUIDE.md)**: Comprehensive ControlNet usage and examples
- **[LoRA Guide](LORA_GUIDE.md)**: LoRA management and best practices
- **[API Reference](API_REFERENCE.md)**: Complete API documentation

## 🚀 Performance & Hardware

### Minimum Requirements
- **RAM**: 8GB system RAM
- **Storage**: 10GB free space
- **Python**: 3.8+

### Recommended Hardware
- **GPU**: 8GB+ VRAM (NVIDIA/AMD)
- **RAM**: 16GB+ system RAM
- **Storage**: SSD with 50GB+ free space

### Supported Platforms
- **CUDA**: NVIDIA GPUs (recommended)
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: All platforms (slower but functional)

## 🔧 Troubleshooting

### Common Issues

#### Slow Startup
If you experience slow startup, ensure you're using the latest version with lazy loading:
```bash
git pull origin main
pip install -e .
```

#### ControlNet Not Working
```bash
# Check preprocessor status
python -c "
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor
print('Available:', controlnet_preprocessor.is_available())
print('Initialized:', controlnet_preprocessor.is_initialized())
"

# Manual initialization
curl -X POST http://localhost:8000/api/controlnet/initialize
```

#### Memory Issues
```bash
# Use smaller image sizes
ollamadiffuser generate "test" --width 512 --height 512

# Enable CPU offloading (automatic)
# Close other applications
# Use basic preprocessors instead of advanced ones
```

### Debug Mode
```bash
# Enable verbose logging
ollamadiffuser -v run model-name

# Check system information
ollamadiffuser info

# Validate installation
ollamadiffuser doctor
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/ollamadiffuser.git
cd ollamadiffuser

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 ollamadiffuser/
black ollamadiffuser/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI**: For Stable Diffusion models
- **Hugging Face**: For model hosting and diffusers library
- **ControlNet Team**: For ControlNet architecture
- **Community**: For feedback and contributions

## 📞 Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ollamadiffuser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ollamadiffuser/discussions)

---

**Happy generating! 🎨✨** 