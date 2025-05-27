#!/usr/bin/env python3
"""
Test script to demonstrate LoRA CLI commands
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return the result"""
    print(f"\n🔧 Running: {cmd}")
    print("=" * 50)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Demonstrate LoRA CLI commands"""
    print("🧪 Testing LoRA CLI Commands\n")
    
    print("📋 Available LoRA Commands:")
    print("=" * 50)
    
    commands = [
        ("Help for LoRA commands", "python -m ollamadiffuser lora --help"),
        ("List installed LoRAs", "python -m ollamadiffuser lora list"),
        ("Pull Ghibli LoRA", "python -m ollamadiffuser lora pull openfree/flux-chatgpt-ghibli-lora --weight-name flux-chatgpt-ghibli-lora.safetensors --alias ghibli"),
        ("Show LoRA info", "python -m ollamadiffuser lora show ghibli"),
        ("Load LoRA (requires model)", "python -m ollamadiffuser lora load ghibli --scale 1.0"),
        ("Unload LoRA", "python -m ollamadiffuser lora unload"),
        ("Remove LoRA", "python -m ollamadiffuser lora rm ghibli"),
    ]
    
    for description, command in commands:
        print(f"\n📝 {description}:")
        print(f"   {command}")
    
    print("\n" + "=" * 70)
    print("🚀 Example Workflow:")
    print("=" * 70)
    
    workflow = [
        "# 1. Download the Ghibli LoRA",
        "python -m ollamadiffuser lora pull openfree/flux-chatgpt-ghibli-lora \\",
        "  --weight-name flux-chatgpt-ghibli-lora.safetensors \\",
        "  --alias ghibli",
        "",
        "# 2. List installed LoRAs",
        "python -m ollamadiffuser lora list",
        "",
        "# 3. Load FLUX model",
        "python -m ollamadiffuser run flux.1-dev",
        "",
        "# 4. In another terminal, load the LoRA",
        "python -m ollamadiffuser lora load ghibli --scale 1.0",
        "",
        "# 5. Generate Ghibli-style images via API",
        "curl -X POST http://localhost:8000/api/generate \\",
        "  -H 'Content-Type: application/json' \\",
        "  -d '{\"prompt\": \"A magical forest in Studio Ghibli style\", \"steps\": 28}'",
        "",
        "# 6. Switch to different LoRA or unload",
        "python -m ollamadiffuser lora unload",
        "python -m ollamadiffuser lora load another_lora --scale 0.8",
    ]
    
    for line in workflow:
        print(line)
    
    print("\n" + "=" * 70)
    print("💡 LoRA Management Tips:")
    print("=" * 70)
    
    tips = [
        "• Use --alias to give LoRAs friendly names (e.g., 'ghibli' instead of 'openfree_flux-chatgpt-ghibli-lora')",
        "• Adjust --scale to control LoRA strength (0.5 = subtle, 1.0 = normal, 1.5 = strong)",
        "• You can load/unload LoRAs without restarting the model",
        "• LoRAs are stored in ~/.ollamadiffuser/loras/",
        "• Use 'lora show <name>' to see detailed information about a LoRA",
        "• Multiple LoRAs can be downloaded but only one loaded at a time",
        "• Some LoRAs work better with specific prompts or styles",
    ]
    
    for tip in tips:
        print(tip)
    
    print("\n" + "=" * 70)
    print("🎨 Popular FLUX LoRAs to try:")
    print("=" * 70)
    
    popular_loras = [
        {
            "name": "Studio Ghibli Style",
            "repo": "openfree/flux-chatgpt-ghibli-lora",
            "weight": "flux-chatgpt-ghibli-lora.safetensors",
            "description": "Creates images in Studio Ghibli animation style"
        },
        {
            "name": "Anime Style",
            "repo": "XLabs-AI/flux-lora-collection",
            "weight": "anime_lora.safetensors",
            "description": "General anime/manga style"
        },
        {
            "name": "Realistic Photos",
            "repo": "XLabs-AI/flux-lora-collection", 
            "weight": "realism_lora.safetensors",
            "description": "Enhanced photorealism"
        }
    ]
    
    for lora in popular_loras:
        print(f"• {lora['name']}")
        print(f"  Repo: {lora['repo']}")
        print(f"  Weight: {lora['weight']}")
        print(f"  Description: {lora['description']}")
        print(f"  Command: python -m ollamadiffuser lora pull {lora['repo']} --weight-name {lora['weight']}")
        print()
    
    print("🎯 Ready to try LoRA management!")
    print("Start with: python -m ollamadiffuser lora --help")

if __name__ == "__main__":
    main() 