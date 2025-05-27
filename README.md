# OllamaDiffuser

🎨 **An Ollama-like image generation model management tool** - Simplify local deployment and management of various image generation models (Stable Diffusion and variants).

## ✨ Features

- 🚀 **One-click Model Management**: Download, run, and switch between different image generation models
- 🔄 **LoRA Support**: Load/unload LoRA adapters with adjustable strength via CLI and Web UI
- 🌐 **Multiple Interfaces**: CLI, REST API, and beautiful Web UI
- 🖥️ **Cross-platform**: Windows, macOS, Linux with automatic hardware optimization
- ⚡ **Hardware Optimization**: Auto-detect and optimize for CUDA, MPS, CPU
- 🎯 **Ollama-style UX**: Familiar command-line experience focused on image generation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ollamadiffuser.git
cd ollamadiffuser

# Quick setup (recommended)
python quick_start.py

# Or manual installation
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```bash
# List available models
ollamadiffuser list

# Download a model (start with smaller ones)
ollamadiffuser pull stable-diffusion-1.5

# Check download status and integrity
ollamadiffuser check stable-diffusion-1.5
ollamadiffuser check --list  # Check all models

# Run model with API server
ollamadiffuser run stable-diffusion-1.5

# Or start Web UI (recommended for beginners)
ollamadiffuser --mode ui
# Visit: http://localhost:8001
```

### Generate Your First Image

**Via Web UI** (Easiest):
1. Run `ollamadiffuser --mode ui`
2. Open http://localhost:8001 in your browser
3. Load a model, enter a prompt, and generate!

**Via API**:
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over mountains"}' \
  --output image.png
```

**Fast Generation with FLUX.1-schnell** (No HuggingFace token required):
```bash
# Download and run FLUX.1-schnell (Apache 2.0 license)
ollamadiffuser pull flux.1-schnell
ollamadiffuser run flux.1-schnell

# Generate high-quality image in ~4 steps (very fast!)
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "width": 1024,
    "height": 1024
  }' \
  --output image.png
```

## 📋 Supported Models

| Model | Size | VRAM | Quality | Speed | License | Best For |
|-------|------|------|---------|-------|---------|----------|
| **Stable Diffusion 1.5** | 5GB | 4GB+ | Good | Fast | CreativeML Open RAIL-M | Learning, quick tests |
| **Stable Diffusion XL** | 7GB | 6GB+ | High | Medium | CreativeML Open RAIL-M | High-quality images |
| **Stable Diffusion 3.5** | 8GB | 8GB+ | Very High | Medium | CreativeML Open RAIL-M | Professional work |
| **FLUX.1-dev** | 15GB | 12GB+ | Excellent | Slow | Non-commercial only | Top-tier quality (research) |
| **FLUX.1-schnell** | 15GB | 12GB+ | Excellent | ⚡ **Very Fast** | ✅ **Apache 2.0** | **Fast production use** |

### Hardware Requirements

**Minimum**: 8GB RAM, 4GB VRAM (or CPU-only)
**Recommended**: 16GB+ RAM, 8GB+ VRAM
**For FLUX**: 24GB+ RAM, 12GB+ VRAM

### 🚀 FLUX Model Comparison

| Aspect | FLUX.1-schnell | FLUX.1-dev |
|--------|----------------|-------------|
| **Speed** | ⚡ 4 steps (12x faster) | 🐌 50 steps |
| **Quality** | 🎯 Excellent | 🎯 Excellent |
| **License** | ✅ Apache 2.0 | ⚠️ Non-commercial only |
| **HF Token** | ❌ Not required | ✅ Required |
| **Commercial Use** | ✅ Allowed | ❌ Not allowed |
| **Guidance Scale** | 0.0 (distilled) | 3.5 (standard) |
| **Best For** | Fast production use | Research/non-commercial |

## 🎯 Command Reference

### Model Management
```bash
ollamadiffuser list                    # List all models
ollamadiffuser list --hardware         # Show hardware requirements
ollamadiffuser pull MODEL_NAME         # Download model
ollamadiffuser check MODEL_NAME        # Check download status and integrity
ollamadiffuser check --list            # List all models with status
ollamadiffuser show MODEL_NAME         # Show model details
ollamadiffuser rm MODEL_NAME           # Remove model
```

### Service Management
```bash
ollamadiffuser run MODEL_NAME          # Run model with API server
ollamadiffuser load MODEL_NAME         # Load model into memory
ollamadiffuser unload                  # Unload current model
ollamadiffuser ps                      # Show running status
ollamadiffuser stop                    # Stop server
```

### Running Modes
```bash
ollamadiffuser --mode cli              # CLI mode (default)
ollamadiffuser --mode api              # API server only
ollamadiffuser --mode ui               # Web UI mode
```

### LoRA Management
```bash
# Download LoRA
ollamadiffuser lora pull REPO_ID --alias NAME

# Load LoRA (requires running model)
ollamadiffuser lora load NAME --scale 1.0

# List and manage LoRAs
ollamadiffuser lora list
ollamadiffuser lora unload
ollamadiffuser lora rm NAME
```

## 🔄 LoRA Usage Guide

LoRAs (Low-Rank Adaptations) allow you to modify model behavior for different styles, faster generation, or specific aesthetics.

### Quick LoRA Workflow

1. **Start a model**:
   ```bash
   ollamadiffuser run stable-diffusion-3.5-medium
   ```

2. **Download LoRA** (in new terminal):
   ```bash
   # Turbo LoRA for faster generation
   ollamadiffuser lora pull tensorart/stable-diffusion-3.5-medium-turbo \
     --weight-name lora_sd3.5m_turbo_8steps.safetensors \
     --alias turbo
   
   # Anime style LoRA
   ollamadiffuser lora pull XLabs-AI/flux-RealismLora \
     --alias realism
   ```

3. **Load LoRA**:
   ```bash
   ollamadiffuser lora load turbo --scale 1.0
   ```

4. **Generate with LoRA**:
   ```bash
   curl -X POST http://localhost:8000/api/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A beautiful landscape", "num_inference_steps": 8}' \
     --output image.png
   ```

### Popular LoRAs

**For FLUX.1-dev**:
- `openfree/flux-chatgpt-ghibli-lora` - Studio Ghibli style
- `XLabs-AI/flux-RealismLora` - Photorealistic enhancement
- `alvdansen/flux-koda` - Kodak film aesthetic

**For SD3.5**:
- `tensorart/stable-diffusion-3.5-medium-turbo` - 8-step fast generation
- `XLabs-AI/sd3-anime-lora` - Anime/manga style

**LoRA Scale Guidelines**:
- `0.5-0.7`: Subtle effect
- `0.8-1.0`: Normal strength (recommended)
- `1.1-1.5`: Strong effect

## 🌐 Web UI Features

The Web UI provides a beautiful, user-friendly interface with:

- 🎨 **Model Management**: Load/unload models with visual status indicators
- 🔄 **LoRA Management**: Download, load, and manage LoRAs with strength control
- 📝 **Image Generation**: Intuitive form with parameter controls
- 📊 **Real-time Status**: Live model and LoRA status indicators
- 🖼️ **Image Display**: Immediate preview of generated images
- 📱 **Responsive Design**: Works on desktop and mobile

Access via: `ollamadiffuser --mode ui` → http://localhost:8001

## 🌐 API Reference

### Image Generation
```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "A beautiful sunset over mountains",
  "negative_prompt": "low quality, blurry",
  "num_inference_steps": 28,
  "guidance_scale": 3.5,
  "width": 1024,
  "height": 1024
}
```

### Model Management
```http
GET /api/models                     # List all models
GET /api/models/running             # Get current model status
POST /api/models/load               # Load model
POST /api/models/unload             # Unload model
```

### LoRA Management
```http
POST /api/lora/load                 # Load LoRA
POST /api/lora/unload               # Unload LoRA
GET /api/lora/status                # Get LoRA status
```

### Health Check
```http
GET /api/health                     # Service health
POST /api/shutdown                  # Graceful shutdown
```

## 🔧 Model-Specific Guides

### FLUX.1-dev Setup

FLUX.1-dev requires HuggingFace access:

1. **Get HuggingFace Token**:
   - Visit [HuggingFace FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
   - Accept license agreement
   - Create access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)

2. **Set Token**:
   ```bash
   export HF_TOKEN=your_token_here
   # or
   huggingface-cli login
   ```

3. **Download and Run**:
   ```bash
   ollamadiffuser pull flux.1-dev
   ollamadiffuser run flux.1-dev
   ```

**FLUX.1-dev Optimal Settings**:
```json
{
  "num_inference_steps": 50,
  "guidance_scale": 3.5,
  "width": 1024,
  "height": 1024,
  "max_sequence_length": 512
}
```

### FLUX.1-schnell Setup

FLUX.1-schnell is the fast, distilled version of FLUX.1-dev with Apache 2.0 license:

1. **Download and Run** (no token required):
   ```bash
   ollamadiffuser pull flux.1-schnell
   ollamadiffuser check flux.1-schnell  # Check download status
   ollamadiffuser run flux.1-schnell
   ```

**FLUX.1-schnell Optimal Settings**:
```json
{
  "num_inference_steps": 4,
  "guidance_scale": 0.0,
  "width": 1024,
  "height": 1024,
  "max_sequence_length": 256
}
```

**Key Benefits**:
- ✅ **No HuggingFace token required** (Apache 2.0 license)
- ✅ **Commercial use allowed**
- ⚡ **12x faster generation** (4 steps vs 50 steps)
- 🎯 **Same quality** as FLUX.1-dev
- 🤖 **Automatic optimization** - engine detects schnell and optimizes parameters
- ⚠️ **No guidance scale** (distilled model - automatically set to 0.0)

**Enhanced Features**:
- **Automatic detection**: Engine automatically optimizes for FLUX.1-schnell
- **Smart parameter adjustment**: Reduces steps to 4 and sets guidance_scale to 0.0
- **Download verification**: Use `ollamadiffuser check flux.1-schnell` for status
- **Universal checker**: Works with all supported models

### Stable Diffusion 3.5

**Optimal Settings**:
```json
{
  "num_inference_steps": 28,
  "guidance_scale": 3.5,
  "width": 1024,
  "height": 1024
}
```

**With Turbo LoRA**:
```json
{
  "num_inference_steps": 8,
  "guidance_scale": 3.5
}
```

## 🛠️ Architecture

```
ollamadiffuser/
├── cli/                 # Command-line interface
├── core/               # Core functionality
│   ├── models/         # Model management
│   ├── inference/      # Inference engines
│   ├── config/         # Configuration
│   └── utils/          # Utilities (LoRA manager, etc.)
├── api/                # REST API server
├── ui/                 # Web interface
│   ├── web.py          # FastAPI app
│   └── templates/      # HTML templates
└── utils/              # Helper scripts
```

## 📦 Dependencies

**Core Requirements**:
- Python 3.8+
- PyTorch 2.0+
- Diffusers 0.21+
- FastAPI 0.100+
- Click 8.0+
- Rich 13.0+

**Hardware Support**:
- NVIDIA CUDA (recommended)
- Apple Metal Performance Shaders (M1/M2)
- CPU fallback (slower)

## 🚨 Troubleshooting

### Common Issues

**"Model doesn't have a device attribute"**:
```bash
pip install -U diffusers transformers
```

**VRAM Out of Memory**:
- Use smaller models (SD 1.5 instead of FLUX)
- Reduce image resolution
- Enable CPU offloading (automatic)

**LoRA Loading Fails**:
- Ensure model is loaded first
- Check LoRA compatibility with current model
- Verify HuggingFace token for gated models

**Slow Generation**:
- Use GPU instead of CPU
- Try Turbo LoRAs for faster generation
- Reduce inference steps

### Performance Tips

1. **Start Small**: Begin with SD 1.5, then upgrade to larger models
2. **Use LoRAs**: Turbo LoRAs can reduce generation time significantly
3. **Batch Generation**: Generate multiple images in sequence for efficiency
4. **Monitor Resources**: Use `ollamadiffuser ps` to check memory usage

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Model Licenses**:
- **Stable Diffusion models**: CreativeML Open RAIL-M
- **FLUX.1-dev**: FLUX.1-dev Non-Commercial License (non-commercial use only)
- **FLUX.1-schnell**: Apache 2.0 (commercial use allowed)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 Links

- [HuggingFace Models](https://huggingface.co/models?pipeline_tag=text-to-image)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA Collections](https://huggingface.co/models?other=lora)

---

**Happy generating!** 🎨✨ 