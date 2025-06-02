# HiDream-I1 Support in OllamaDiffuser

## Overview

OllamaDiffuser now supports the HiDream-I1 series of state-of-the-art image generation models from HiDream.ai. These are 17B parameter models that deliver exceptional image quality with commercial-friendly MIT licensing.

## Supported Models

Three HiDream-I1 variants are now available:

### HiDream-I1-Dev (`hidream-i1-dev`)
- **Best for**: Balanced performance and quality
- **Inference Steps**: 28 (optimal)
- **Guidance Scale**: 0.0 (distilled model)
- **Hardware**: 24GB+ VRAM, 32GB+ RAM recommended
- **License**: MIT (commercial use allowed)

### HiDream-I1-Full (`hidream-i1-full`)
- **Best for**: Highest quality generation
- **Inference Steps**: 50 (optimal)
- **Guidance Scale**: 5.0
- **Hardware**: 24GB+ VRAM, 32GB+ RAM recommended
- **License**: MIT (commercial use allowed)

### HiDream-I1-Fast (`hidream-i1-fast`)
- **Best for**: Fast generation with good quality
- **Inference Steps**: 16 (optimal)
- **Guidance Scale**: 0.0 (distilled model)
- **Hardware**: 24GB+ VRAM, 32GB+ RAM recommended
- **License**: MIT (commercial use allowed)

## Features

✅ **Complete Integration**: Full support in CLI, web UI, and Python API  
✅ **Hardware Optimization**: Automatic device-specific optimizations for CUDA, MPS, and CPU  
✅ **Parameter Handling**: Model-specific parameter optimization  
✅ **Error Handling**: Comprehensive error handling and fallbacks  
✅ **License Support**: MIT license with commercial use permissions  

## Prerequisites

### 1. Update Diffusers
HiDream models require the latest development version of diffusers:

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### 2. Llama Model Access
HiDream models require access to Meta's Llama-3.1-8B-Instruct model for text encoding:

```bash
# Login with HuggingFace CLI
huggingface-cli login

# Accept the Llama license at:
# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

### 3. Hardware Requirements
- **Minimum**: 24GB VRAM, 32GB RAM
- **Recommended**: 32GB+ VRAM, 64GB+ RAM
- **Supported Devices**: NVIDIA CUDA, Apple Silicon (MPS)
- **Optimal GPUs**: RTX 4080+, RTX 4090, or Apple M3 Max

## Usage

### CLI Usage

```bash
# List available models (includes HiDream models)
python -m ollamadiffuser list

# View detailed hardware requirements
python -m ollamadiffuser list --hardware

# Download a HiDream model (requires significant disk space and time)
python -m ollamadiffuser pull hidream-i1-dev

# Switch to HiDream model
python -m ollamadiffuser use hidream-i1-dev

# Generate image with HiDream
python -m ollamadiffuser generate "a beautiful landscape with mountains and lakes"

# Generate with custom parameters
python -m ollamadiffuser generate "a futuristic city" --steps 28 --size 1024x1024
```

### Python API Usage

```python
from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.inference.engine import InferenceEngine

# Check available HiDream models
hidream_models = [m for m in model_manager.list_available_models() if 'hidream' in m]
print("Available HiDream models:", hidream_models)

# Download and load HiDream-I1-Dev
if model_manager.pull_model("hidream-i1-dev"):
    if model_manager.load_model("hidream-i1-dev"):
        engine = model_manager.loaded_model
        
        # Generate image
        image = engine.generate_image(
            prompt="a serene Japanese garden with cherry blossoms",
            width=1024,
            height=1024,
            num_inference_steps=28  # Optimal for HiDream-I1-Dev
        )
        
        # Save image
        image.save("hidream_output.png")
        print("Image generated successfully!")
```

### Web UI Usage

1. Start the web server:
   ```bash
   python -m ollamadiffuser serve
   ```

2. Open browser to `http://localhost:8000`

3. Use the model management interface to download HiDream models

4. Select HiDream model from the dropdown and generate images

## Performance Notes

### Model-Specific Optimizations

- **HiDream-I1-Dev/Fast**: Automatically uses guidance_scale=0.0 (distilled models)
- **HiDream-I1-Full**: Uses guidance_scale=5.0 for best quality
- **Max Sequence Length**: Automatically set to 512 for optimal text encoding
- **Device Optimization**: Automatic dtype selection (bfloat16 for GPU, float32 for CPU)

### Platform-Specific Behavior

**NVIDIA CUDA**:
- Uses bfloat16 precision for optimal performance
- Enables memory optimizations and attention slicing
- Best performance platform for HiDream models

**Apple Silicon (MPS)**:
- Automatic optimization for Apple GPUs
- Memory-efficient processing
- Good performance on M3 Max and newer

**CPU**:
- Falls back to float32 for stability
- Reduces inference steps for acceptable performance
- Includes performance warnings about expected slow generation

## Troubleshooting

### Common Issues

1. **"HiDreamImagePipeline not available"**
   ```bash
   pip install git+https://github.com/huggingface/diffusers.git
   ```

2. **"Failed to load Llama components"**
   - Ensure you're logged into HuggingFace: `huggingface-cli login`
   - Accept Llama license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

3. **Out of Memory Errors**
   - Ensure you have 24GB+ VRAM
   - Try CPU inference (slower but requires less VRAM)
   - Close other GPU-intensive applications

4. **Slow Generation**
   - Expected on CPU (several minutes per image)
   - Use GPU for better performance
   - Consider using HiDream-I1-Fast for quicker results

### Performance Tips

- **Use appropriate variant**: Fast for speed, Dev for balance, Full for quality
- **Optimal sizes**: 1024x1024 works best for HiDream models
- **Batch generation**: Generate multiple images in sequence for efficiency
- **Memory management**: Unload models between sessions if needed

## Technical Details

### Architecture
- **Parameters**: 17B total parameters
- **Text Encoder**: Meta-Llama-3.1-8B-Instruct
- **Pipeline**: HiDreamImagePipeline (diffusers)
- **Precision**: bfloat16 (GPU), float32 (CPU)

### Benchmark Performance
- **HPSv2.1**: 33.82 (state-of-the-art)
- **GenEval**: 0.83 (excellent)
- **DPG-Bench**: 85.89 (superior)

## License

HiDream-I1 models are released under the **MIT License**, making them suitable for:
- ✅ Commercial use
- ✅ Distribution
- ✅ Modification
- ✅ Private use

No special licensing agreements or restrictions apply.

## Support

For issues specific to HiDream model integration:
1. Check this documentation first
2. Verify prerequisites are met
3. Check hardware requirements
4. Open an issue with detailed error logs

For HiDream model questions:
- Visit: https://huggingface.co/HiDream-ai
- Original paper and documentation available on HuggingFace model pages 