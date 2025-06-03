# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-06-02

### 🚀 Major Features Added

#### ⚡ GGUF Model Support
- **Quantized Models**: Full support for GGUF (GPT-Generated Unified Format) quantized models
- **Massive VRAM Reduction**: Run FLUX.1-dev with 3GB VRAM instead of 20GB+ 
- **7 Quantization Levels**: From q2k (3GB) to f16 (16GB) for different hardware capabilities
- **Hardware Optimization**: Native CUDA and Metal acceleration support
- **CPU Fallback**: Automatic CPU inference when VRAM is insufficient

#### 🎛️ GGUF Model Variants
- **flux.1-dev-gguf-q2k**: Ultra-low VRAM (3GB) for testing and low-end hardware
- **flux.1-dev-gguf-q3ks**: Balanced option (4GB) for mobile GPUs
- **flux.1-dev-gguf-q4ks**: **Recommended** (6GB) - best quality/performance balance
- **flux.1-dev-gguf-q5ks**: High quality (8GB) for mid-range GPUs
- **flux.1-dev-gguf-q6k**: Near-original quality (10GB) 
- **flux.1-dev-gguf-q8**: Minimal quality loss (12GB)
- **flux.1-dev-gguf-f16**: Full precision (16GB)

### 🛠️ Technical Implementation

#### GGUF Engine Integration
- **Backend**: stable-diffusion.cpp with Python bindings integration
- **Automatic Detection**: Seamless GGUF model recognition and loading
- **Memory Management**: Intelligent VRAM usage and CPU offloading
- **Hardware Acceleration**: CMAKE-based CUDA and Metal compilation support

#### CLI Enhancements
- **GGUF Check**: `ollamadiffuser registry check-gguf` command for compatibility verification
- **Model Pull**: Seamless GGUF model downloading with progress tracking
- **Status Monitoring**: Real-time GGUF support and model status checking

### 🎯 Performance Optimizations

#### Generation Parameters
- **Optimized Settings**: 4-step generation (FLUX-optimized)
- **CFG Scale**: guidance_scale=1.0 for best FLUX results
- **Euler Sampler**: Recommended sampler for GGUF models
- **Hardware Adaptation**: Automatic parameter adjustment based on available VRAM

#### Memory Efficiency
- **Smart Loading**: Load only required model components
- **Progressive Quantization**: Automatic fallback to lower quantization when needed
- **Resource Management**: Intelligent GPU memory allocation and cleanup

### 📚 Documentation & Guides

#### Comprehensive GGUF Guide
- **GGUF_GUIDE.md**: Complete 160+ line guide with installation, usage, and troubleshooting
- **Hardware Recommendations**: Specific guidance for different GPU tiers
- **Performance Comparisons**: Quality vs speed vs VRAM usage tables
- **Troubleshooting**: Common issues and solutions for GGUF models

#### Usage Examples
- **CLI Workflows**: Step-by-step GGUF model usage examples
- **Python API**: Code examples for programmatic GGUF model usage
- **Web UI Integration**: Browser-based GGUF model selection and generation

### 🔧 Dependencies & Requirements

#### New Dependencies
- **stable-diffusion-cpp-python**: Core GGUF inference engine
- **gguf**: Model format handling and validation
- **Enhanced OpenCV**: Updated to >=4.8.0 for improved compatibility

#### Hardware Support
- **NVIDIA CUDA**: CMAKE_ARGS="-DSD_CUDA=ON" installation
- **Apple Metal**: CMAKE_ARGS="-DSD_METAL=ON" for M1/M2 Macs
- **CPU Inference**: Full CPU fallback support for any modern processor

### 🎨 User Experience Improvements

#### Accessibility
- **Low-End Hardware**: Enable FLUX.1-dev on 3GB GPUs (previously impossible)
- **Faster Downloads**: Reduced model sizes from ~24GB to 3-16GB
- **Quick Testing**: Instant model switching between quantization levels

#### Web UI Enhancements
- **GGUF Model Selection**: Dropdown menu with GGUF model variants
- **VRAM Monitoring**: Real-time memory usage display
- **Quality Preview**: Visual quality indicators for each quantization level

### 🐛 Bug Fixes & Improvements
- **Memory Leaks**: Improved GGUF model cleanup and resource management
- **Error Handling**: Better error messages for GGUF-specific issues
- **Compatibility**: Enhanced hardware detection and fallback mechanisms

### ⚠️ Breaking Changes
- **Dependency Requirements**: New GGUF dependencies required for full functionality
- **Model Loading**: GGUF models use different loading mechanisms than regular models

### 🔄 Migration Guide
For users upgrading to v1.2.0:

1. **Install GGUF Dependencies**: `pip install stable-diffusion-cpp-python gguf`
2. **Check Compatibility**: `ollamadiffuser registry check-gguf`
3. **Download GGUF Model**: `ollamadiffuser pull flux.1-dev-gguf-q4ks`
4. **Update Hardware Acceleration**: Reinstall with CUDA/Metal support if needed

### 📊 Performance Metrics
- **VRAM Reduction**: Up to 85% reduction (20GB → 3GB)
- **File Size**: Up to 87% smaller downloads (24GB → 3GB)
- **Generation Speed**: Comparable or faster due to optimized quantization
- **Quality Retention**: 90%+ quality retention with q4ks quantization

## [1.1.6] - 2025-5-30

### 🎨 New Features

#### ControlNet Sample Images
- **New CLI Command**: `ollamadiffuser create-samples` for creating ControlNet demonstration images
- **Built-in Samples**: Pre-made control images for Canny, Depth, OpenPose, and Scribble controls
- **Web UI Integration**: Sample images automatically available in the web interface for easy testing
- **Force Recreation**: `--force` flag to recreate all samples even if they exist

#### Installation Helper
- **New Script**: `install_helper.py` for platform-specific installation guidance
- **Shell Detection**: Automatically detects user's shell (bash, zsh, fish) and provides correct install syntax
- **Multiple Installation Options**: Basic, Full, and Development installation commands
- **Interactive Installation**: Option to install directly from the helper script

### 🛠️ Improvements

#### CLI Enhancements
- **Progress Tracking**: Enhanced download progress display with Ollama-style formatting
- **Better Error Handling**: Improved error messages and graceful failure modes
- **Warning Fixes**: Resolved various CLI warnings and edge cases

#### Web UI Enhancements
- **Sample Image Gallery**: Built-in ControlNet samples with 3 images per control type
- **Automatic Sample Creation**: Samples generated automatically when needed
- **Better UX**: Visual samples make ControlNet testing more intuitive

### 🐛 Bug Fixes
- **Version Inconsistencies**: Fixed version numbering across different components
- **Installation Issues**: Resolved shell-specific installation syntax problems
- **CLI Warnings**: Fixed various warning messages and edge cases
- **Sample Generation**: Improved reliability of sample image creation

### 📦 Technical Changes
- **MANIFEST.in**: Updated to include sample images and static files
- **Dependencies**: Refined dependency management for better compatibility
- **Shell Compatibility**: Better support for zsh, fish, and bash shells

### 🎯 Sample Images Created
- **Canny Control**: Geometric shapes, house outline, portrait silhouette (3 samples)
- **Depth Control**: Depth map variations for different scene types (3 samples)
- **OpenPose Control**: Human pose variations for different positions (3 samples)
- **Scribble Control**: Hand-drawn style sketches and outlines (3 samples)

## [1.1.0] - 2025-5-29

### 🚀 Major Features Added

#### ⚡ Lazy Loading Architecture
- **Instant Startup**: Application now starts immediately without downloading ControlNet models
- **On-Demand Loading**: ControlNet preprocessors initialize only when actually needed
- **Performance Boost**: `ollamadiffuser --help` runs in milliseconds instead of 30+ seconds
- **Memory Efficient**: No unnecessary model downloads for users who don't use ControlNet

#### 🎛️ Complete ControlNet Integration
- **6 ControlNet Models**: SD 1.5 and SDXL variants (canny, depth, openpose, scribble)
- **10 Control Types**: canny, depth, openpose, hed, mlsd, normal, lineart, lineart_anime, shuffle, scribble
- **Advanced Preprocessors**: Full controlnet-aux integration with graceful fallbacks
- **Web UI Integration**: File upload, preprocessing, and side-by-side result display
- **REST API Support**: Complete API endpoints for ControlNet generation and preprocessing

#### 🔄 Enhanced LoRA Management
- **Web UI Integration**: Download LoRAs directly from Hugging Face in the browser
- **Alias Support**: Create custom names for your LoRAs
- **Strength Control**: Adjust LoRA influence with intuitive sliders
- **Real-time Loading**: Load/unload LoRAs without restarting the application

### 🛠️ Technical Improvements

#### ControlNet Preprocessor Manager
- **Lazy Initialization**: `ControlNetPreprocessorManager` with `is_initialized()`, `is_available()`, `initialize()` methods
- **Automatic Fallback**: Basic OpenCV processors when advanced ones fail
- **Error Handling**: Robust validation and graceful degradation
- **Status Tracking**: Real-time initialization and availability status

#### Web UI Enhancements
- **ControlNet Section**: Dedicated controls with status indicators
- **Initialization Button**: Manual preprocessor initialization for faster processing
- **File Upload**: Drag-and-drop control image upload with validation
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **Real-time Status**: Live model, LoRA, and ControlNet status indicators

#### API Improvements
- **New Endpoints**: `/api/controlnet/initialize`, `/api/controlnet/preprocessors`, `/api/controlnet/preprocess`
- **File Upload Support**: Multipart form data handling for control images
- **Status Endpoints**: Check ControlNet availability and initialization status
- **Error Handling**: Comprehensive error responses with helpful messages

### 📦 Dependencies Updated
- **controlnet-aux**: Added `>=0.0.7` for advanced preprocessing capabilities
- **opencv-python**: Added `>=4.8.0` for basic image processing fallbacks
- **diffusers**: Updated to `>=0.26.0` for ControlNet compatibility

### 🎨 User Experience Improvements

#### Startup Performance
- **Before**: 30+ seconds startup time, 1GB+ automatic downloads
- **After**: Instant startup, downloads only when needed
- **User Control**: Choose when to initialize ControlNet preprocessors

#### Web UI Experience
- **Status Indicators**: Clear visual feedback for all system states
- **Progressive Loading**: Initialize components as needed
- **Error Messages**: Helpful guidance for common issues
- **Mobile Support**: Responsive design works on all devices

#### CLI Experience
- **Fast Commands**: All CLI commands run instantly
- **Lazy Loading**: ControlNet models load only when generating
- **Status Commands**: Check system state without triggering downloads

### 🔧 Configuration Changes
- **setup.py**: Added ControlNet dependencies
- **pyproject.toml**: Updated dependency specifications
- **Model Registry**: Enhanced with ControlNet model definitions

### 📚 Documentation Updates
- **CONTROLNET_GUIDE.md**: Comprehensive 400+ line guide with examples
- **README.md**: Updated with lazy loading features and ControlNet quick start
- **API Documentation**: Complete endpoint reference with examples

### 🐛 Bug Fixes
- **Startup Crashes**: Fixed 404 errors from non-existent model repositories
- **Memory Leaks**: Improved cleanup of ControlNet preprocessors
- **Device Compatibility**: Better handling of CPU/GPU device switching
- **Error Handling**: More graceful failure modes with helpful messages

### ⚠️ Breaking Changes
- **Import Behavior**: `controlnet_preprocessors` module no longer auto-initializes
- **API Changes**: Some ControlNet endpoints require explicit initialization

### 🔄 Migration Guide
For users upgrading from v1.0.x:

1. **No Action Required**: Lazy loading is automatic and transparent
2. **Web UI**: ControlNet preprocessors initialize automatically when uploading images
3. **API Users**: Call `/api/controlnet/initialize` for faster subsequent processing
4. **Python API**: Use `controlnet_preprocessor.initialize()` for batch processing

### 🎯 Performance Metrics
- **Startup Time**: Reduced from 30+ seconds to <1 second
- **Memory Usage**: Reduced baseline memory footprint by ~2GB
- **First Generation**: Slightly slower due to lazy loading, then normal speed
- **Subsequent Generations**: Same performance as before

## [1.0.0] - 2025-5-28

### Added
- Initial release with core functionality
- Support for Stable Diffusion 1.5, SDXL, SD3, and FLUX models
- Basic LoRA support
- CLI interface
- REST API server
- Web UI interface
- Model management system

### Features
- Model downloading and management
- Image generation with various parameters
- Multiple interface options (CLI, API, Web UI)
- Hardware optimization (CUDA, MPS, CPU)
- Safety checker bypass for creative freedom

---

## Development Notes

### Version Numbering
- **Major** (X.0.0): Breaking changes, major feature additions
- **Minor** (1.X.0): New features, significant improvements
- **Patch** (1.1.X): Bug fixes, minor improvements

### Release Process
1. Update version in `__init__.py`
2. Update CHANGELOG.md with new features
3. Update documentation
4. Create release tag
5. Deploy to package repositories 