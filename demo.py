#!/usr/bin/env python3
"""
OllamaDiffuser demo script
Demonstrates how to use programmatic access to model management and image generation functionality
"""

import asyncio
import requests
import json
from pathlib import Path

# Import OllamaDiffuser components
from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.config.settings import settings

def demo_model_management():
    """Demonstrate model management functionality"""
    print("🔧 OllamaDiffuser Model Management Demo")
    print("=" * 50)
    
    # List available models
    print("\n📋 Available models:")
    available_models = model_manager.list_available_models()
    for model in available_models:
        print(f"  • {model}")
    
    # List installed models
    print("\n✅ Installed models:")
    installed_models = model_manager.list_installed_models()
    if installed_models:
        for model in installed_models:
            print(f"  • {model}")
    else:
        print("  (None)")
    
    # Get model information
    print("\n🔍 Model details:")
    for model_name in available_models[:2]:  # Only show first two
        info = model_manager.get_model_info(model_name)
        if info:
            print(f"  {model_name}:")
            print(f"    Type: {info.get('model_type')}")
            print(f"    Variant: {info.get('variant')}")
            print(f"    Installed: {info.get('installed', False)}")

def demo_api_client():
    """Demonstrate API client functionality"""
    print("\n🌐 API Client Demo")
    print("=" * 50)
    
    base_url = f"http://{settings.server.host}:{settings.server.port}"
    
    try:
        # Check health status
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API server connection successful")
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   Current model: {health_data.get('current_model', 'None')}")
        else:
            print("❌ API server connection failed")
            return
        
        # Get model list
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            models_data = response.json()
            print(f"\n📋 Available models: {len(models_data.get('available', []))}")
            print(f"   Installed models: {len(models_data.get('installed', []))}")
        
        # If model is loaded, try to generate image
        if health_data.get('model_loaded', False):
            print("\n🎨 Attempting to generate image...")
            generate_data = {
                "prompt": "A beautiful sunset over mountains",
                "negative_prompt": "low quality, blurry",
                "num_inference_steps": 4,  # Use fewer steps to save time
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512
            }
            
            response = requests.post(
                f"{base_url}/api/generate",
                json=generate_data,
                timeout=120
            )
            
            if response.status_code == 200:
                # Save image
                output_path = Path("demo_output.png")
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Image generation successful, saved to: {output_path}")
            else:
                print(f"❌ Image generation failed: {response.status_code}")
        else:
            print("⚠️  No model loaded, skipping image generation demo")
            
    except requests.exceptions.ConnectionError:
        print("❌ Unable to connect to API server")
        print("   Please start the server first: ollamadiffuser serve")
    except Exception as e:
        print(f"❌ API demo error: {e}")

def print_usage_examples():
    """Print usage examples"""
    print("\n📚 Usage Examples")
    print("=" * 50)
    
    examples = [
        ("List all available models", "ollamadiffuser list"),
        ("Download model", "ollamadiffuser pull stable-diffusion-3.5-medium"),
        ("Run model service", "ollamadiffuser run stable-diffusion-3.5-medium"),
        ("Start API server", "ollamadiffuser serve"),
        ("Start Web UI", "python -m ollamadiffuser --mode ui"),
        ("Show model information", "ollamadiffuser show stable-diffusion-3.5-medium"),
        ("Unload model", "ollamadiffuser unload"),
        ("Remove model", "ollamadiffuser rm stable-diffusion-3.5-medium"),
    ]
    
    for desc, cmd in examples:
        print(f"  {desc}:")
        print(f"    {cmd}")
        print()

def main():
    """Main demo function"""
    print("🎨 OllamaDiffuser Demo")
    print("An Ollama-like image generation model management tool")
    print("=" * 60)
    
    # Display configuration information
    print(f"\n⚙️  Configuration:")
    print(f"   Config directory: {settings.config_dir}")
    print(f"   Models directory: {settings.models_dir}")
    print(f"   Server address: {settings.server.host}:{settings.server.port}")
    
    # Demonstrate model management
    demo_model_management()
    
    # Demonstrate API client
    demo_api_client()
    
    # Print usage examples
    print_usage_examples()
    
    print("🎉 Demo completed!")
    print("\nQuick start:")
    print("1. Download model: ollamadiffuser pull stable-diffusion-3.5-medium")
    print("2. Run service: ollamadiffuser run stable-diffusion-3.5-medium")
    print("3. Access Web UI: python -m ollamadiffuser --mode ui")

if __name__ == "__main__":
    main() 