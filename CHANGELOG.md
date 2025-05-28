# Changelog

All notable changes to OllamaDiffuser will be documented in this file.

## [1.0.0] - 2024-12-XX

### ✨ Added
- **Web UI Mode**: Beautiful web interface with `ollamadiffuser --mode ui`
- **LoRA Support**: Complete LoRA management via CLI and Web UI
  - Download LoRAs from HuggingFace Hub
  - Load/unload LoRAs with adjustable strength
  - Real-time LoRA status indicators
- **Multiple Running Modes**: CLI, API server, and Web UI modes
- **Model Support**: 
  - Stable Diffusion 1.5
  - Stable Diffusion XL
  - Stable Diffusion 3.5 Medium
  - FLUX.1-dev (with HuggingFace token support)
  - FLUX.1-schnell (with HuggingFace token support)
- **Hardware Optimization**: Auto-detection for CUDA, MPS, and CPU
- **Comprehensive CLI**: Ollama-style commands for model management
- **REST API**: Full API for integration with other applications
- **Cross-platform**: Windows, macOS, and Linux support

### 🔧 Technical Features
- FastAPI-based web server
- Jinja2 templating for Web UI
- Rich CLI with beautiful output
- Automatic memory optimization
- Progress tracking for downloads
- Configuration management
- Error handling and recovery

### 📚 Documentation
- Comprehensive README with all guides merged
- LoRA usage examples
- Model-specific setup instructions
- API documentation
- Troubleshooting guide

### 🧹 Project Cleanup
- Organized project structure
- Moved examples to dedicated directory
- Consolidated documentation
- Improved .gitignore
- Better package metadata

## [Unreleased]

### 🚀 Planned Features
- ControlNet support
- Image-to-image generation
- Batch generation
- Model quantization
- Plugin system
- Docker support

---

## Version Format

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible) 