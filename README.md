# OllamaDiffuser 🎨

[![PyPI version](https://badge.fury.io/py/ollamadiffuser.svg)](https://badge.fury.io/py/ollamadiffuser)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


## Local AI Image Generation with OllamaDiffuser

**OllamaDiffuser** simplifies local deployment of **Stable Diffusion**, **FLUX.1**, and other AI image generation models. An intuitive **local SD** tool inspired by **Ollama's** simplicity - perfect for **local diffuser** workflows with CLI, web UI, and LoRA support.

🌐 **Website**: [ollamadiffuser.com](https://www.ollamadiffuser.com/) | 📦 **PyPI**: [pypi.org/project/ollamadiffuser](https://pypi.org/project/ollamadiffuser/)

---

## ✨ Features

- **🚀 Fast Startup**: Instant application launch with lazy loading architecture
- **🎛️ ControlNet Support**: Precise image generation control with 10+ control types
- **🔄 LoRA Integration**: Dynamic LoRA loading and management
- **📦 GGUF Support**: Memory-efficient quantized models (3GB VRAM minimum!)
- **🌐 Multiple Interfaces**: CLI, Python API, Web UI, and REST API
- **📦 Model Management**: Easy installation and switching between models
- **⚡ Performance Optimized**: Memory-efficient with GPU acceleration
- **🎨 Professional Results**: High-quality image generation with fine-tuned control

## 🚀 Quick Start

### Option 1: Install from PyPI (Recommended)
```bash
# Install from PyPI
pip install ollamadiffuser

# Pull and run a model (4-command setup)
ollamadiffuser pull flux.1-schnell
ollamadiffuser run flux.1-schnell

# Generate via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset"}' \
  --output image.png
```

### GGUF Quick Start (Low VRAM)
```bash
# For systems with limited VRAM (3GB+)
pip install ollamadiffuser stable-diffusion-cpp-python gguf

# Download memory-efficient GGUF model
ollamadiffuser pull flux.1-dev-gguf-q4ks

# Generate with reduced memory usage
ollamadiffuser run flux.1-dev-gguf-q4ks
```

### Option 2: Development Installation
```bash
# Clone the repository
git clone https://github.com/ollamadiffuser/ollamadiffuser.git
cd ollamadiffuser

# Install dependencies
pip install -e .
```

### Basic Usage
```bash
# Check version
ollamadiffuser -V

# Install a model
ollamadiffuser pull stable-diffusion-1.5

# Run the model (loads and starts API server)
ollamadiffuser run stable-diffusion-1.5

# Generate an image via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains"}' \
  --output image.png

# Start web interface
ollamadiffuser --mode ui

open http://localhost:8001
```

### ControlNet Quick Start
```bash
# Install ControlNet model
ollamadiffuser pull controlnet-canny-sd15

# Run ControlNet model (loads and starts API server)
ollamadiffuser run controlnet-canny-sd15

# Generate with control image
curl -X POST http://localhost:8000/api/generate/controlnet \
  -F "prompt=a beautiful landscape" \
  -F "control_image=@your_image.jpg"
```

---

## 🎯 Supported Models

Choose from a variety of state-of-the-art image generation models:

| Model | License | Quality | Speed | Commercial Use | VRAM |
|-------|---------|---------|-------|----------------|------|
| **FLUX.1-schnell** | Apache 2.0 | High | **4 steps** (12x faster) | ✅ Commercial OK | 20GB+ |
| **FLUX.1-dev** | Non-commercial | High | 50 steps | ❌ Non-commercial | 20GB+ |
| **FLUX.1-dev-gguf** | Non-commercial | High | 4 steps | ❌ Non-commercial | **3-16GB** |
| **Stable Diffusion 3.5** | CreativeML | Medium | 28 steps | ⚠️ Check License | 12GB+ |
| **Stable Diffusion 1.5** | CreativeML | Fast | Lightweight | ⚠️ Check License | 6GB+ |

### 💾 GGUF Models - Reduced Memory Requirements

**NEW**: GGUF quantized models enable running FLUX.1-dev on budget hardware!

| GGUF Variant | VRAM | Quality | Best For |
|--------------|------|---------|----------|
| `flux.1-dev-gguf-q4ks` | 6GB | ⭐⭐⭐⭐ | **Recommended** - RTX 3060/4060 |
| `flux.1-dev-gguf-q3ks` | 4GB | ⭐⭐⭐ | Mobile GPUs, GTX 1660 Ti |
| `flux.1-dev-gguf-q2k` | 3GB | ⭐⭐ | Entry-level hardware |
| `flux.1-dev-gguf-q6k` | 10GB | ⭐⭐⭐⭐⭐ | RTX 3080/4070+ |

📖 **[Complete GGUF Guide](GGUF_GUIDE.md)** - Hardware recommendations, installation, and optimization tips

### Why Choose FLUX.1-schnell?
- **Apache 2.0 license** - Perfect for commercial use
- **4-step generation** - Lightning fast results  
- **Commercial OK** - Use in your business

---

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
# Pull and run a model
ollamadiffuser pull stable-diffusion-1.5
ollamadiffuser run stable-diffusion-1.5

# In another terminal, generate images via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic cityscape",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024
  }' \
  --output image.png
```

### Web UI
```bash
# Start web interface
ollamadiffuser --mode ui
Open http://localhost:8001
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

ollamadiffuser load stable-diffusion-1.5

# Generate image
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape", "width": 1024, "height": 1024}'

# API document
http://localhost:8000/docs
```

### Python API
```python
from ollamadiffuser.core.models.manager import model_manager

# Load model
success = model_manager.load_model("stable-diffusion-1.5")
if success:
    engine = model_manager.loaded_model
    
    # Generate image
    image = engine.generate_image(
        prompt="a beautiful sunset",
        width=1024,
        height=1024
    )
    image.save("output.jpg")
else:
    print("Failed to load model")
```

## 📦 Supported Models

### Base Models
- **Stable Diffusion 1.5**: Classic, reliable, fast
- **Stable Diffusion XL**: High-resolution, detailed
- **Stable Diffusion 3**: Latest architecture
- **FLUX.1**: State-of-the-art quality

### GGUF Quantized Models
- **FLUX.1-dev GGUF**: 7 quantization levels (3GB-16GB VRAM)
- **Memory Efficient**: Run high-quality models on budget hardware
- **Same API**: Works seamlessly with existing commands

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

### GGUF Model Usage
```bash
# Check GGUF support
ollamadiffuser registry check-gguf

# Download GGUF model for your hardware
ollamadiffuser pull flux.1-dev-gguf-q4ks  # 6GB VRAM
ollamadiffuser pull flux.1-dev-gguf-q3ks  # 4GB VRAM

# Use with optimized settings
ollamadiffuser run flux.1-dev-gguf-q4ks
```

### Batch Processing
```python
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor

# Pre-initialize for faster processing
controlnet_preprocessor.initialize()

# Process multiple images
prompt = "beautiful landscape"  # Define the prompt
for i, image_path in enumerate(image_list):
    control_img = controlnet_preprocessor.preprocess(image_path, "canny")
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

## 📚 Documentation & Guides

- **[GGUF Models Guide](GGUF_GUIDE.md)**: Complete guide to memory-efficient GGUF models
- **[ControlNet Guide](CONTROLNET_GUIDE.md)**: Comprehensive ControlNet usage and examples
- **[Installation Guide](INSTALLATION_GUIDE.md)**: Detailed installation instructions
- **[Website Documentation](https://www.ollamadiffuser.com/)**: Complete tutorials and guides

## 🚀 Performance & Hardware

### Minimum Requirements
- **RAM**: 8GB system RAM
- **Storage**: 10GB free space
- **Python**: 3.8+

### Recommended Hardware

#### For Regular Models
- **GPU**: 8GB+ VRAM (NVIDIA/AMD)
- **RAM**: 16GB+ system RAM
- **Storage**: SSD with 50GB+ free space

#### For GGUF Models (Memory Efficient)
- **GPU**: 3GB+ VRAM (or CPU only)
- **RAM**: 8GB+ system RAM (16GB+ for CPU inference)
- **Storage**: SSD with 20GB+ free space

### Supported Platforms
- **CUDA**: NVIDIA GPUs (recommended)
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: All platforms (slower but functional)

## 🔧 Troubleshooting

### Installation Issues

#### Missing Dependencies (cv2/OpenCV Error)
If you encounter `ModuleNotFoundError: No module named 'cv2'`, run:

```bash
# Quick fix
pip install opencv-python>=4.8.0

# Or use the built-in verification tool
ollamadiffuser verify-deps

# Or install with all optional dependencies
# For bash/sh:
pip install ollamadiffuser[full]

# For zsh (macOS default):
pip install "ollamadiffuser[full]"

# For fish shell:
pip install 'ollamadiffuser[full]'
```

#### GGUF Support Issues
```bash
# Install GGUF dependencies
pip install stable-diffusion-cpp-python gguf

# Check GGUF support
ollamadiffuser registry check-gguf

# See full GGUF troubleshooting guide
# Read GGUF_GUIDE.md for detailed troubleshooting
```

#### Complete Dependency Check
```bash
# Run comprehensive system diagnostics
ollamadiffuser doctor

# Verify and install missing dependencies interactively
ollamadiffuser verify-deps
```

#### Clean Installation
If you're having persistent issues:

```bash
# Uninstall and reinstall
pip uninstall ollamadiffuser

# Reinstall with all dependencies (shell-specific syntax):
# For bash/sh:
pip install --no-cache-dir ollamadiffuser[full]

# For zsh (macOS default):
pip install --no-cache-dir "ollamadiffuser[full]"

# For fish shell:
pip install --no-cache-dir 'ollamadiffuser[full]'

# Verify installation
ollamadiffuser verify-deps
```

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
# Use GGUF models for lower memory usage
ollamadiffuser pull flux.1-dev-gguf-q4ks  # 6GB VRAM
ollamadiffuser pull flux.1-dev-gguf-q3ks  # 4GB VRAM

# Use smaller image sizes via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "width": 512, "height": 512}' \
  --output test.png

# CPU offloading is automatic
# Close other applications to free memory
# Use basic preprocessors instead of advanced ones
```

### Platform-Specific Issues

#### macOS Apple Silicon
```bash
# If you encounter OpenCV issues on Apple Silicon
pip uninstall opencv-python
pip install opencv-python-headless>=4.8.0

# For GGUF Metal acceleration
CMAKE_ARGS="-DSD_METAL=ON" pip install stable-diffusion-cpp-python
```

#### Windows
```bash
# If you encounter build errors
pip install --only-binary=all opencv-python>=4.8.0

# For GGUF CUDA acceleration
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

#### Linux
```bash
# If you need system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
pip install opencv-python>=4.8.0
```

### Debug Mode
```bash
# Enable verbose logging
ollamadiffuser --verbose run model-name
```

## 🤝 Contributing

We welcome contributions! Please check the GitHub repository for contribution guidelines.

## 🤝 Community & Support

### Quick Actions

- **🐛 [Report a Bug](https://github.com/ollamadiffuser/ollamadiffuser/issues)** - Found an issue? Let us know
- **💡 [Feature Request](https://github.com/ollamadiffuser/ollamadiffuser/issues)** - Have an idea? Share it with us  
- **💬 [Join Discussions](https://github.com/ollamadiffuser/ollamadiffuser/discussions)** - Community discussion
- **⭐ [Star on GitHub](https://github.com/ollamadiffuser/ollamadiffuser)** - Show your support

### Community Driven

OllamaDiffuser is an open-source project that thrives on community feedback. Every suggestion, bug report, and contribution helps make it better for everyone.

**Open Source** • **Community Driven** • **Actively Maintained**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI**: For Stable Diffusion models
- **Black Forest Labs**: For FLUX.1 models
- **city96**: For FLUX.1-dev GGUF quantizations
- **Hugging Face**: For model hosting and diffusers library
- **ControlNet Team**: For ControlNet architecture
- **Community**: For feedback and contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ollamadiffuser/ollamadiffuser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ollamadiffuser/ollamadiffuser/discussions)

---

**Ready to get started?** Install from PyPI: `pip install ollamadiffuser` or visit [ollamadiffuser.com](https://www.ollamadiffuser.com/) 🎨✨ 