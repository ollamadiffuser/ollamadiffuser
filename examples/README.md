# Examples and Test Scripts

This directory contains example scripts, test files, and utilities for OllamaDiffuser.

## 🚀 Getting Started

### Quick Start Script
- **`quick_start.py`** - Automated setup and installation script

### Test Scripts
- **`test_installation.py`** - Verify installation and dependencies
- **`test_flux_support.py`** - Test FLUX.1-dev model support
- **`test_flux_ghibli_lora.py`** - Test FLUX with Ghibli LoRA
- **`test_lora_cli.py`** - Test LoRA CLI commands
- **`test_device_fix.py`** - Test device compatibility fixes
- **`test_fastapi_server.py`** - Test FastAPI server functionality

### Demo Scripts
- **`demo.py`** - Interactive demonstration of features

### Utility Scripts
- **`monitor_download_progress.py`** - Monitor model download progress
- **`check_flux_download.py`** - Check FLUX model download status
- **`server_legacy.py`** - Legacy server implementation

## Usage

```bash
# Run quick setup
python examples/quick_start.py

# Test installation
python examples/test_installation.py

# Run demo
python examples/demo.py

# Test specific functionality
python examples/test_flux_support.py
```

## Note

These scripts are for testing and demonstration purposes. For production use, use the main CLI commands:

```bash
ollamadiffuser --help
ollamadiffuser --mode ui
``` 