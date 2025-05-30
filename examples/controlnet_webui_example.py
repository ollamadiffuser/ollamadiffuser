#!/usr/bin/env python3
"""
ControlNet Web UI Example for OllamaDiffuser

This example demonstrates the new lazy loading ControlNet features in the Web UI.
Shows how ControlNet preprocessors initialize on-demand for optimal performance.

Features demonstrated:
- Lazy loading architecture (instant startup)
- Automatic ControlNet preprocessor initialization
- Manual initialization for faster processing
- Real-time status indicators
- File upload and preprocessing
- Side-by-side result display

Usage:
    python examples/controlnet_webui_example.py
    
Then open http://localhost:8001 in your browser.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor
from ollamadiffuser.ui.web import create_ui_app
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_lazy_loading():
    """Demonstrate the lazy loading features"""
    print("\n🚀 OllamaDiffuser ControlNet Web UI Example")
    print("=" * 50)
    
    print("\n⚡ Lazy Loading Demonstration:")
    print("-" * 30)
    
    # Show that preprocessors are not initialized at startup
    print(f"📊 ControlNet Available: {controlnet_preprocessor.is_available()}")
    print(f"📊 ControlNet Initialized: {controlnet_preprocessor.is_initialized()}")
    print(f"📊 Available Types: {controlnet_preprocessor.get_available_types()}")
    
    print("\n✨ Key Benefits:")
    print("  • Instant startup - no waiting for model downloads")
    print("  • Memory efficient - only loads when needed")
    print("  • User choice - initialize manually or automatically")
    print("  • Graceful fallback - basic processors if advanced ones fail")
    
    print("\n🎛️ ControlNet Features in Web UI:")
    print("  • Real-time status indicators")
    print("  • Automatic initialization when uploading images")
    print("  • Manual initialization button for faster processing")
    print("  • Side-by-side control and generated image display")
    print("  • Responsive design for desktop and mobile")
    
    return True

def check_models():
    """Check if required models are available"""
    print("\n📦 Checking Available Models:")
    print("-" * 30)
    
    available_models = model_manager.list_available_models()
    installed_models = model_manager.list_installed_models()
    
    print(f"📋 Available Models: {len(available_models)}")
    for model in available_models:
        status = "✅ Installed" if model in installed_models else "❌ Not Installed"
        print(f"  • {model}: {status}")
    
    # Check for ControlNet models specifically
    controlnet_models = [m for m in available_models if 'controlnet' in m]
    if controlnet_models:
        print(f"\n🎛️ ControlNet Models Available: {len(controlnet_models)}")
        for model in controlnet_models:
            status = "✅ Installed" if model in installed_models else "❌ Not Installed"
            print(f"  • {model}: {status}")
    else:
        print("\n⚠️  No ControlNet models found in registry")
    
    if not installed_models:
        print("\n💡 To install models, run:")
        print("   ollamadiffuser pull stable-diffusion-1.5")
        print("   ollamadiffuser pull controlnet-canny-sd15")
    
    return len(installed_models) > 0

def demonstrate_initialization():
    """Demonstrate manual initialization"""
    print("\n🔧 Manual Initialization Example:")
    print("-" * 30)
    
    if not controlnet_preprocessor.is_available():
        print("❌ ControlNet preprocessors not available")
        print("   Install with: pip install controlnet-aux>=0.0.7")
        return False
    
    if controlnet_preprocessor.is_initialized():
        print("✅ ControlNet preprocessors already initialized")
        return True
    
    print("🚀 Initializing ControlNet preprocessors...")
    start_time = time.time()
    
    success = controlnet_preprocessor.initialize()
    init_time = time.time() - start_time
    
    if success:
        print(f"✅ Initialization successful in {init_time:.2f} seconds")
        print(f"📊 Available types: {controlnet_preprocessor.get_available_types()}")
        return True
    else:
        print("❌ Initialization failed - will use basic fallback processors")
        return False

def create_example_app():
    """Create the Web UI app with example configuration"""
    print("\n🌐 Creating Web UI Application:")
    print("-" * 30)
    
    # Create the FastAPI app
    app = create_ui_app()
    
    print("✅ Web UI application created")
    print("🎨 Features enabled:")
    print("  • Model management with status indicators")
    print("  • LoRA download and management")
    print("  • ControlNet integration with lazy loading")
    print("  • Responsive design for all devices")
    print("  • Real-time status updates")
    
    return app

def print_usage_instructions():
    """Print instructions for using the Web UI"""
    print("\n📖 Web UI Usage Instructions:")
    print("=" * 50)
    
    print("\n1️⃣ Model Management:")
    print("   • Load a model from the dropdown (required for generation)")
    print("   • For ControlNet: Load a ControlNet model (e.g., controlnet-canny-sd15)")
    print("   • Status indicators show model loading state")
    
    print("\n2️⃣ LoRA Management:")
    print("   • Download LoRAs by entering Hugging Face repository ID")
    print("   • Load LoRAs with adjustable strength (0.1-2.0)")
    print("   • Unload LoRAs without restarting the application")
    
    print("\n3️⃣ ControlNet Features:")
    print("   • Upload control images for precise generation control")
    print("   • Preprocessors initialize automatically when uploading")
    print("   • Manual initialization available for faster processing")
    print("   • Adjust conditioning scale and guidance timing")
    
    print("\n4️⃣ Image Generation:")
    print("   • Enter descriptive prompts for best results")
    print("   • Use negative prompts to avoid unwanted elements")
    print("   • Adjust parameters like steps and guidance scale")
    print("   • View results with side-by-side control image display")
    
    print("\n🎯 Tips for Best Results:")
    print("   • Start with basic models (SD 1.5) before trying larger ones")
    print("   • Use clear, high-contrast control images")
    print("   • Experiment with conditioning scale (0.8-1.2 usually works well)")
    print("   • Try different control types for different use cases")

def main():
    """Main function to run the ControlNet Web UI example"""
    try:
        # Demonstrate lazy loading
        demonstrate_lazy_loading()
        
        # Check available models
        models_available = check_models()
        
        # Demonstrate manual initialization (optional)
        print("\n🤔 Would you like to pre-initialize ControlNet preprocessors?")
        print("   (They will initialize automatically when needed, but this makes it faster)")
        response = input("   Initialize now? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            demonstrate_initialization()
        else:
            print("✅ Skipping initialization - will happen automatically when needed")
        
        # Create the Web UI app
        app = create_example_app()
        
        # Print usage instructions
        print_usage_instructions()
        
        # Start the server
        print("\n🚀 Starting Web UI Server:")
        print("-" * 30)
        print("🌐 URL: http://localhost:8001")
        print("📱 Mobile-friendly responsive design")
        print("⚡ Lazy loading for optimal performance")
        print("\n💡 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Web UI server...")
        print("Thank you for trying the ControlNet Web UI example!")
        
    except Exception as e:
        logger.error(f"Error running Web UI example: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure all dependencies are installed: pip install -e .")
        print("   • Check that port 8001 is available")
        print("   • Try running with verbose logging: python -v examples/controlnet_webui_example.py")

if __name__ == "__main__":
    main() 