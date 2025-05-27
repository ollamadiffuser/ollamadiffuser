#!/usr/bin/env python3
"""
Check model download status and provide resume options for any supported model
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.config.settings import settings
from ollamadiffuser.core.utils.download_utils import check_download_integrity, get_repo_file_list, format_size

def check_download_status(model_name: str):
    """Check the current download status of any model"""
    print(f"🔍 Checking {model_name} download status...\n")
    
    # Check if model is in registry
    if model_name not in model_manager.model_registry:
        print(f"❌ {model_name} not found in model registry")
        available_models = model_manager.list_available_models()
        print(f"📋 Available models: {', '.join(available_models)}")
        return False
    
    model_info = model_manager.model_registry[model_name]
    repo_id = model_info["repo_id"]
    model_path = settings.get_model_path(model_name)
    
    print(f"📦 Model: {model_name}")
    print(f"🔗 Repository: {repo_id}")
    print(f"📁 Local path: {model_path}")
    
    # Show model-specific info
    license_info = model_info.get("license_info", {})
    if license_info:
        print(f"📄 License: {license_info.get('type', 'Unknown')}")
        print(f"🔑 HF Token Required: {'Yes' if license_info.get('requires_agreement', False) else 'No'}")
        print(f"💼 Commercial Use: {'Allowed' if license_info.get('commercial_use', False) else 'Not Allowed'}")
    
    # Show optimal parameters
    params = model_info.get("parameters", {})
    if params:
        print(f"⚡ Optimal Settings:")
        print(f"   Steps: {params.get('num_inference_steps', 'N/A')}")
        print(f"   Guidance: {params.get('guidance_scale', 'N/A')}")
        if 'max_sequence_length' in params:
            print(f"   Max Seq Length: {params['max_sequence_length']}")
    
    print()
    
    # Check if directory exists
    if not model_path.exists():
        print("📂 Status: Not downloaded")
        return False
    
    # Get repository file list
    print("🌐 Getting repository information...")
    try:
        file_sizes = get_repo_file_list(repo_id)
        total_expected_size = sum(file_sizes.values())
        total_files_expected = len(file_sizes)
        
        print(f"📊 Expected: {total_files_expected} files, {format_size(total_expected_size)} total")
    except Exception as e:
        print(f"⚠️ Could not get repository info: {e}")
        file_sizes = {}
        total_expected_size = 0
        total_files_expected = 0
    
    # Check local files
    local_files = []
    local_size = 0
    
    for file_path in model_path.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(model_path)
            file_size = file_path.stat().st_size
            local_files.append((str(rel_path), file_size))
            local_size += file_size
    
    print(f"💾 Downloaded: {len(local_files)} files, {format_size(local_size)} total")
    
    if total_expected_size > 0:
        progress_percent = (local_size / total_expected_size) * 100
        print(f"📈 Progress: {progress_percent:.1f}%")
    
    print()
    
    # Check for missing files
    if file_sizes:
        # Check if we have size information from the API
        has_size_info = any(size > 0 for size in file_sizes.values())
        
        if has_size_info:
            # Normal case: we have size information, do detailed comparison
            missing_files = []
            incomplete_files = []
            
            for expected_file, expected_size in file_sizes.items():
                local_file_path = model_path / expected_file
                if not local_file_path.exists():
                    missing_files.append(expected_file)
                elif expected_size > 0 and local_file_path.stat().st_size != expected_size:
                    local_size_actual = local_file_path.stat().st_size
                    incomplete_files.append((expected_file, local_size_actual, expected_size))
            
            if missing_files:
                print(f"❌ Missing files ({len(missing_files)}):")
                for missing_file in missing_files[:10]:  # Show first 10
                    print(f"   - {missing_file}")
                if len(missing_files) > 10:
                    print(f"   ... and {len(missing_files) - 10} more")
                print()
            
            if incomplete_files:
                print(f"⚠️ Incomplete files ({len(incomplete_files)}):")
                for incomplete_file, actual_size, expected_size in incomplete_files[:5]:
                    print(f"   - {incomplete_file}: {format_size(actual_size)}/{format_size(expected_size)}")
                if len(incomplete_files) > 5:
                    print(f"   ... and {len(incomplete_files) - 5} more")
                print()
            
            if not missing_files and not incomplete_files:
                print("✅ All files present and complete!")
                
                # Check integrity
                print("🔍 Checking download integrity...")
                if check_download_integrity(str(model_path), repo_id):
                    print("✅ Download integrity verified!")
                    
                    # Check if model is in configuration
                    if model_manager.is_model_installed(model_name):
                        print("✅ Model is properly configured")
                        return True
                    else:
                        print("⚠️ Model files complete but not in configuration")
                        return "needs_config"
                else:
                    print("❌ Download integrity check failed")
                    return False
            else:
                print("⚠️ Download is incomplete")
                return "incomplete"
        else:
            # No size information available from API (common with gated repos)
            print("ℹ️ Repository API doesn't provide file sizes (common with gated models)")
            print("🔍 Checking essential model files instead...")
            
            # Check for essential model files
            essential_files = ['model_index.json']
            essential_dirs = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'scheduler']
            
            missing_essential = []
            for essential_file in essential_files:
                if not (model_path / essential_file).exists():
                    missing_essential.append(essential_file)
            
            existing_dirs = []
            for essential_dir in essential_dirs:
                if (model_path / essential_dir).exists():
                    existing_dirs.append(essential_dir)
            
            if missing_essential:
                print(f"❌ Missing essential files: {', '.join(missing_essential)}")
                return "incomplete"
            
            if existing_dirs:
                print(f"✅ Found model components: {', '.join(existing_dirs)}")
            
            # Check integrity
            print("🔍 Checking download integrity...")
            if check_download_integrity(str(model_path), repo_id):
                print("✅ Download integrity verified!")
                
                # Check if model is in configuration
                if model_manager.is_model_installed(model_name):
                    print("✅ Model is properly configured and functional")
                    return True
                else:
                    print("⚠️ Model files complete but not in configuration")
                    return "needs_config"
            else:
                print("❌ Download integrity check failed")
                return False
    
    # Check if download process is running
    print("🔍 Checking for active download processes...")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if f'ollamadiffuser pull {model_name}' in result.stdout:
            print("🔄 Download process is currently running")
            return "downloading"
        else:
            print("💤 No active download process found")
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
    
    return "incomplete"

def show_model_specific_help(model_name: str):
    """Show model-specific help and recommendations"""
    model_info = model_manager.get_model_info(model_name)
    if not model_info:
        return
    
    print(f"\n💡 {model_name} Specific Tips:")
    
    # License-specific help
    license_info = model_info.get("license_info", {})
    if license_info.get("requires_agreement", False):
        print(f"   🔑 Requires HuggingFace token and license agreement")
        print(f"   📝 Visit: https://huggingface.co/{model_info['repo_id']}")
        print(f"   🔧 Set token: export HF_TOKEN=your_token_here")
    else:
        print(f"   ✅ No HuggingFace token required!")
    
    # Model-specific optimizations
    if "schnell" in model_name.lower():
        print(f"   ⚡ FLUX.1-schnell is 12x faster than FLUX.1-dev")
        print(f"   🎯 Optimized for 4-step generation")
        print(f"   💼 Commercial use allowed (Apache 2.0)")
    elif "flux.1-dev" in model_name.lower():
        print(f"   🎨 Best quality FLUX model")
        print(f"   🔬 Requires 50 steps for optimal results")
        print(f"   ⚠️ Non-commercial license only")
    elif "stable-diffusion-1.5" in model_name.lower():
        print(f"   🚀 Great for learning and quick tests")
        print(f"   💾 Smallest model, runs on most hardware")
    elif "stable-diffusion-3.5" in model_name.lower():
        print(f"   🏆 Excellent quality-to-speed ratio")
        print(f"   🔄 Great LoRA ecosystem")
    
    # Hardware recommendations
    hw_req = model_info.get("hardware_requirements", {})
    if hw_req:
        min_vram = hw_req.get("min_vram_gb", 0)
        if min_vram >= 12:
            print(f"   🖥️ Requires high-end GPU (RTX 4070+ or M2 Pro+)")
        elif min_vram >= 8:
            print(f"   🖥️ Requires mid-range GPU (RTX 3080+ or M1 Pro+)")
        else:
            print(f"   🖥️ Runs on most modern GPUs")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check model download status and provide resume options")
    parser.add_argument("model_name", nargs="?", default=None,
                       help="Model name to check (e.g., flux.1-schnell, flux.1-dev, stable-diffusion-1.5)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available models")
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Available Models:")
        available_models = model_manager.list_available_models()
        for model in available_models:
            model_info = model_manager.get_model_info(model)
            status = "✅ Installed" if model_manager.is_model_installed(model) else "⬇️ Available"
            license_type = model_info.get("license_info", {}).get("type", "Unknown")
            print(f"   {model:<30} {status:<15} ({license_type})")
        return
    
    if not args.model_name:
        print("🧪 Model Download Status Checker\n")
        print("Usage: python check_model_download.py MODEL_NAME")
        print("       python check_model_download.py --list")
        print("\nExamples:")
        print("   python check_model_download.py flux.1-schnell")
        print("   python check_model_download.py flux.1-dev")
        print("   python check_model_download.py stable-diffusion-1.5")
        return
    
    model_name = args.model_name
    print(f"🧪 {model_name} Download Status Checker\n")
    
    status = check_download_status(model_name)
    
    print("\n" + "="*60)
    
    if status is True:
        print(f"🎉 {model_name} is ready to use!")
        print(f"\n💡 You can now run:")
        print(f"   ollamadiffuser run {model_name}")
    elif status == "needs_config":
        print(f"⚠️ {model_name} files are complete but model needs configuration")
        print(f"\n💡 Try reinstalling:")
        print(f"   ollamadiffuser pull {model_name} --force")
    elif status == "downloading":
        print(f"🔄 {model_name} is currently downloading")
        print(f"\n💡 Wait for download to complete or check progress")
    elif status == "incomplete":
        print(f"⚠️ Download is incomplete")
        print(f"\n💡 Resume download with:")
        print(f"   ollamadiffuser pull {model_name}")
        print(f"\n💡 Or force fresh download with:")
        print(f"   ollamadiffuser pull {model_name} --force")
    else:
        print(f"❌ {model_name} is not downloaded")
        print(f"\n💡 Download with:")
        print(f"   ollamadiffuser pull {model_name}")
    
    show_model_specific_help(model_name)
    
    print(f"\n📚 For more help: ollamadiffuser --help")

if __name__ == "__main__":
    main() 