# Stable Diffusion 3.5 LoRA Usage Guide

## 🎨 **Loading LoRAs for SD3.5**

The LoRA system works identically for all models (FLUX, SD3.5, SDXL, etc.). Here's how to use LoRAs with Stable Diffusion 3.5:

## **Step-by-Step Workflow**

### **1. Start SD3.5 Model**

```bash
# Load the SD3.5 model with server
python -m ollamadiffuser run stable-diffusion-3.5-medium
```

**Important**: Keep this terminal running! The model is loaded in this process.

### **2. Download SD3.5 Compatible LoRAs**

#### **Popular SD3.5 LoRAs:**

```bash
# SD3.5 Turbo LoRA (faster generation)
python -m ollamadiffuser lora pull tensorart/stable-diffusion-3.5-medium-turbo \
  --weight-name lora_sd3.5m_turbo_8steps.safetensors \
  --alias sd35-turbo

# Anime Style LoRA for SD3.5
python -m ollamadiffuser lora pull XLabs-AI/sd3-anime-lora \
  --weight-name sd3_anime.safetensors \
  --alias sd35-anime

# Realistic Photo LoRA for SD3.5
python -m ollamadiffuser lora pull stabilityai/sd3-lora-realism \
  --weight-name realism_lora.safetensors \
  --alias sd35-realism
```

### **3. Load LoRA into SD3.5**

**Method A: Via API (Recommended)**
```bash
# In a NEW terminal, load LoRA via API
curl -X POST http://localhost:8000/api/lora/load \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "sd35-turbo",
    "repo_id": "tensorart/stable-diffusion-3.5-medium-turbo",
    "weight_name": "lora_sd3.5m_turbo_8steps.safetensors",
    "scale": 1.0
  }'
```

**Method B: Via CLI (Alternative)**
```bash
# Only works if model is loaded in CLI process (not server)
python -m ollamadiffuser lora load sd35-turbo --scale 1.0
```

### **4. Generate Images with LoRA**

#### **Via API:**

```bash
# Generate with Turbo LoRA (fewer steps needed)
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains and lakes",
    "num_inference_steps": 8,
    "guidance_scale": 3.5,
    "width": 1024,
    "height": 1024
  }' \
  --output landscape_turbo.png

# Generate with Anime LoRA
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute anime girl with blue hair in a magical forest",
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "width": 1024,
    "height": 1024
  }' \
  --output anime_girl.png
```

#### **Via Python:**

```python
import requests
import json
from PIL import Image
import io

# Generate with loaded LoRA
response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "prompt": "A photorealistic portrait of a wise wizard",
        "negative_prompt": "low quality, blurry, distorted",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024
    }
)

if response.status_code == 200:
    image = Image.open(io.BytesIO(response.content))
    image.save("wizard_sd35.png")
    print("Image saved!")
```

### **5. Switch Between LoRAs**

```bash
# Unload current LoRA
python -m ollamadiffuser lora unload

# Load different LoRA
python -m ollamadiffuser lora load sd35-anime --scale 0.9

# Or load multiple LoRAs (if supported)
python -m ollamadiffuser lora load sd35-turbo --scale 0.5
```

## **🔧 LoRA Management Commands**

### **List Available LoRAs:**

```bash
python -m ollamadiffuser lora list
```

### **Show LoRA Details:**

```bash
python -m ollamadiffuser lora show sd35-turbo
```

### **Remove LoRA:**

```bash
python -m ollamadiffuser lora rm sd35-turbo
```

## **⚡ SD3.5 Specific Tips**

### **1. Turbo LoRA Benefits:**
- **Faster Generation**: Reduces steps from 28 to 8
- **Same Quality**: Maintains SD3.5 quality
- **Better Performance**: Ideal for real-time applications

### **2. Optimal Parameters for SD3.5:**

```json
{
  "num_inference_steps": 28,     // Standard: 28, Turbo: 8
  "guidance_scale": 3.5,         // SD3.5 works well with 3.0-4.0
  "width": 1024,                 // SD3.5 native resolution
  "height": 1024,
  "negative_prompt": "low quality, bad anatomy, worst quality"
}
```

### **3. LoRA Scale Guidelines:**

- **0.5-0.7**: Subtle effect
- **0.8-1.0**: Normal strength (recommended)
- **1.1-1.5**: Strong effect
- **1.6+**: May cause artifacts

## **🎯 Popular SD3.5 LoRA Repositories**

### **Speed Enhancement:**
```bash
# Turbo LoRA (8 steps instead of 28)
python -m ollamadiffuser lora pull tensorart/stable-diffusion-3.5-medium-turbo \
  --weight-name lora_sd3.5m_turbo_8steps.safetensors \
  --alias turbo
```

### **Style LoRAs:**
```bash
# Anime/Manga style
python -m ollamadiffuser lora pull XLabs-AI/sd3-anime-collection \
  --weight-name anime_lora.safetensors \
  --alias anime

# Photorealistic enhancement
python -m ollamadiffuser lora pull stabilityai/sd3-photo-lora \
  --weight-name photo_realism.safetensors \
  --alias photo

# Artistic styles
python -m ollamadiffuser lora pull huggingface/sd3-art-styles \
  --weight-name impressionist.safetensors \
  --alias impressionist
```

### **Character LoRAs:**
```bash
# Specific character styles
python -m ollamadiffuser lora pull civitai/sd3-character-pack \
  --weight-name character_lora.safetensors \
  --alias character
```

## **🚀 Complete Example Workflow**

```bash
# 1. Start SD3.5
python -m ollamadiffuser run stable-diffusion-3.5-medium

# 2. Download Turbo LoRA for speed
python -m ollamadiffuser lora pull tensorart/stable-diffusion-3.5-medium-turbo \
  --weight-name lora_sd3.5m_turbo_8steps.safetensors \
  --alias turbo

# 3. Load the LoRA
python -m ollamadiffuser lora load turbo --scale 1.0

# 4. Generate fast images (8 steps instead of 28)
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic dragon flying over a medieval castle",
    "num_inference_steps": 8,
    "guidance_scale": 3.5,
    "width": 1024,
    "height": 1024
  }' \
  --output dragon_fast.png

# 5. Switch to anime style
python -m ollamadiffuser lora unload
python -m ollamadiffuser lora pull XLabs-AI/sd3-anime-lora \
  --weight-name anime.safetensors \
  --alias anime
python -m ollamadiffuser lora load anime --scale 0.8

# 6. Generate anime-style image
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute anime girl with pink hair in a cherry blossom garden",
    "num_inference_steps": 28,
    "guidance_scale": 3.5
  }' \
  --output anime_girl.png
```

## **💡 Pro Tips**

### **1. Performance Optimization:**
- Use **Turbo LoRA** for faster generation
- Lower **guidance_scale** (3.0-3.5) for SD3.5
- Use **1024x1024** for best quality

### **2. Quality Enhancement:**
- Combine **style LoRAs** with **quality LoRAs**
- Use **negative prompts** effectively
- Experiment with **LoRA scales**

### **3. Workflow Efficiency:**
- Keep multiple LoRAs downloaded
- Use **aliases** for easy switching
- Test different **scales** for optimal results

## **🔍 Troubleshooting**

### **LoRA Not Loading:**
```bash
# Check if model is running
python -m ollamadiffuser ps

# Check LoRA status
python -m ollamadiffuser lora list

# Verify LoRA files
python -m ollamadiffuser lora show <lora_name>
```

### **Generation Issues:**
- **Too fast/low quality**: Increase inference steps
- **Too slow**: Use Turbo LoRA or reduce steps
- **Artifacts**: Lower LoRA scale or guidance scale
- **Not enough style**: Increase LoRA scale

### **Memory Issues:**
- **VRAM shortage**: Use CPU offloading (automatic)
- **RAM shortage**: Close other applications
- **Disk space**: Remove unused LoRAs

## **🎉 Ready to Create Amazing SD3.5 Images!**

The LoRA system gives you complete control over SD3.5 styling and performance. Experiment with different LoRAs and scales to find your perfect workflow!

---

**Note**: SD3.5 LoRAs are specifically trained for the SD3.5 architecture and may not work with other models like FLUX or SDXL. Always use model-specific LoRAs for best results. 