# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-12-XX

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

## [1.0.0] - 2024-11-XX

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