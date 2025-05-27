# FLUX.1-dev 使用指南

## 概述

FLUX.1-dev 是 Black Forest Labs 开发的12B参数高质量文本到图像生成模型，现已在 OllamaDiffuser 中得到支持。

## 特性

- **高质量输出**：仅次于 FLUX.1 [pro] 的顶级图像质量
- **优秀的提示跟随**：与闭源替代方案相匹配的性能
- **引导蒸馏训练**：使 FLUX.1-dev 更加高效
- **开放权重**：推动新的科学研究，赋能艺术家开发创新工作流程

## 系统要求

### 硬件要求
- **最低 VRAM**: 12GB
- **推荐 VRAM**: 16GB
- **最低 RAM**: 24GB
- **推荐 RAM**: 32GB
- **磁盘空间**: 15GB
- **支持设备**: NVIDIA CUDA, Apple MPS (M1/M2)

### 推荐硬件
- **NVIDIA**: RTX 4070 或更高
- **Apple**: M2 Pro 或更高

## 前置要求

### 1. HuggingFace 账户和 Token

FLUX.1-dev 需要 HuggingFace 账户和访问权限：

1. 访问 [HuggingFace](https://huggingface.co) 创建账户
2. 访问 [FLUX.1-dev 模型页面](https://huggingface.co/black-forest-labs/FLUX.1-dev)
3. 同意许可协议（FLUX.1-dev Non-Commercial License）
4. 创建访问 Token：
   - 访问 [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - 创建新的 Token（需要 "Read" 权限）

### 2. 设置 HuggingFace Token

```bash
# 方法1：环境变量
export HF_TOKEN=your_token_here

# 方法2：通过 huggingface-hub 登录
huggingface-cli login
```

## 安装和使用

### 1. 检查支持

```bash
# 运行测试脚本检查 FLUX.1-dev 支持
python test_flux_support.py
```

### 2. 查看可用模型

```bash
# 列出所有可用模型
ollamadiffuser list

# 查看详细硬件要求
ollamadiffuser list --hardware
```

### 3. 下载模型

```bash
# 下载 FLUX.1-dev 模型（约15GB）
ollamadiffuser pull flux.1-dev
```

### 4. 运行模型

```bash
# 启动 FLUX.1-dev 服务
ollamadiffuser run flux.1-dev
```

### 5. 生成图像

#### 通过 API

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic dragon soaring through clouds at sunset",
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
    "width": 1024,
    "height": 1024
  }' \
  --output dragon.png
```

#### 通过 Python

```python
import requests
import json
from PIL import Image
import io

# 生成请求
response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "prompt": "A beautiful landscape with mountains and a lake",
        "negative_prompt": "low quality, blurry, distorted",
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024
    }
)

# 保存图像
if response.status_code == 200:
    image = Image.open(io.BytesIO(response.content))
    image.save("landscape.png")
    print("图像已保存为 landscape.png")
```

#### 通过 Web UI

```bash
# 启动 Web UI
python -m ollamadiffuser --mode ui

# 访问 http://localhost:8001
```

## 优化建议

### 内存优化

FLUX.1-dev 会自动启用以下优化：

- **Attention Slicing**: 减少内存使用
- **CPU Offloading**: 将部分模型移至 CPU 以节省 VRAM
- **bfloat16 精度**: 平衡质量和性能

### 生成参数调优

```json
{
  "prompt": "your prompt here",
  "negative_prompt": "low quality, bad anatomy, worst quality, low resolution",
  "num_inference_steps": 50,        // 推荐 28-50 步
  "guidance_scale": 3.5,            // 推荐 3.0-4.0
  "width": 1024,                    // 支持多种分辨率
  "height": 1024,
  "max_sequence_length": 512        // FLUX 特有参数
}
```

### 性能提示

1. **首次加载较慢**：模型较大，首次加载需要时间
2. **推理速度**：在推荐硬件上，1024x1024 图像约需 30-60 秒
3. **批量生成**：考虑使用较少的推理步数进行快速预览

## 许可证和使用限制

### FLUX.1-dev Non-Commercial License

- ✅ **个人使用**：允许个人、科学和研究用途
- ✅ **开源项目**：允许开源和非商业项目
- ❌ **商业用途**：不允许商业用途
- ❌ **有害内容**：禁止生成有害或非法内容

### 使用限制

模型不得用于：
- 违反法律法规的用途
- 伤害未成年人
- 生成虚假信息
- 骚扰或威胁他人
- 生成非法色情内容
- 完全自动化的决策系统

## 故障排除

### 常见问题

#### 1. "Model doesn't have a device attribute" 错误
```bash
# 更新 diffusers 库
pip install -U diffusers
```

#### 2. VRAM 不足
```bash
# 使用 CPU offloading（自动启用）
# 或减少生成分辨率
```

#### 3. 下载失败
```bash
# 检查 HuggingFace token
echo $HF_TOKEN

# 重新登录
huggingface-cli login
```

#### 4. 生成速度慢
- 确保使用 GPU（CUDA 或 MPS）
- 减少推理步数（28-35 步）
- 使用较小的分辨率进行测试

### 性能基准

| 硬件 | 分辨率 | 步数 | 大约时间 |
|------|--------|------|----------|
| RTX 4090 | 1024x1024 | 50 | 25-35s |
| RTX 4070 | 1024x1024 | 50 | 45-60s |
| M2 Pro | 1024x1024 | 50 | 60-90s |
| M1 Pro | 1024x1024 | 50 | 90-120s |

## 示例提示

### 高质量提示示例

```
"A photorealistic portrait of a wise old wizard with a long white beard, wearing intricate robes, standing in a mystical library filled with floating books and glowing orbs, dramatic lighting, highly detailed, 8k resolution"

"A futuristic cityscape at night with neon lights reflecting on wet streets, flying cars in the sky, cyberpunk aesthetic, detailed architecture, atmospheric fog, cinematic composition"

"A serene Japanese garden in autumn with a traditional wooden bridge over a koi pond, maple trees with red and orange leaves, soft morning light filtering through the trees, peaceful atmosphere"
```

### 负面提示建议

```
"low quality, bad anatomy, worst quality, low resolution, blurry, distorted, deformed, ugly, poorly drawn, bad proportions, extra limbs, missing limbs, bad hands, bad fingers, watermark, signature, text"
```

## 更多资源

- [FLUX.1-dev 官方页面](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Black Forest Labs 博客](https://blackforestlabs.ai/)
- [Diffusers 文档](https://huggingface.co/docs/diffusers/)
- [OllamaDiffuser GitHub](https://github.com/your-username/ollamadiffuser)

---

**注意**: FLUX.1-dev 是一个强大的模型，请负责任地使用，遵守许可协议和使用条款。 

## ✅ **LoRA CLI Commands Successfully Implemented!**

### 🎯 **What We've Built:**

1. **Complete LoRA Management System** with CLI commands:
   - `lora pull` - Download LoRA weights from Hugging Face Hub
   - `lora list` - List installed LoRA weights
   - `lora show` - Show detailed LoRA information
   - `lora load` - Load LoRA into current model
   - `lora unload` - Unload current LoRA
   - `lora rm` - Remove LoRA weights

2. **Enhanced FLUX Model** with automatic Ghibli LoRA support
3. **Fixed Device Detection** for MPS/CPU compatibility
4. **Robust Download System** with progress tracking and resume capability

### 🚀 **Available Commands:**

```bash
# Download LoRA with friendly alias
python -m ollamadiffuser lora pull openfree/flux-chatgpt-ghibli-lora \
  --weight-name flux-chatgpt-ghibli-lora.safetensors \
  --alias ghibli

# List installed LoRAs
python -m ollamadiffuser lora list

# Show detailed LoRA info
python -m ollamadiffuser lora show ghibli

# Load LoRA into model (requires model to be running)
python -m ollamadiffuser lora load ghibli --scale 1.0

# Unload LoRA
python -m ollamadiffuser lora unload

# Remove LoRA
python -m ollamadiffuser lora rm ghibli
```

### 🎨 **Example Workflow:**

1. **Download the Ghibli LoRA** (✅ Working!)
2. **Start FLUX model**: `python -m ollamadiffuser run flux.1-dev`
3. **Load LoRA**: `python -m ollamadiffuser lora load ghibli --scale 1.0`
4. **Generate Ghibli-style images** via API or web interface
5. **Switch LoRAs** without restarting the model

### 💡 **Key Features:**

- **Friendly aliases** for LoRAs (e.g., "ghibli" instead of long repo names)
- **Scale control** (0.5 = subtle, 1.0 = normal, 1.5 = strong)
- **Runtime loading/unloading** without model restart
- **Progress tracking** with detailed download status
- **Automatic resume** for interrupted downloads
- **Rich CLI interface** with tables and colors

The LoRA management system is now fully functional! You can download, manage, and use LoRA weights with FLUX models easily through the command line. The system is designed to be user-friendly and robust, with proper error handling and progress feedback.