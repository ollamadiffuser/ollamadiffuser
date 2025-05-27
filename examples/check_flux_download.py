#!/usr/bin/env python3
"""
Check FLUX.1-dev download status and provide resume options
"""

import os
import sys
from pathlib import Path
import subprocess
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.config.settings import settings
from ollamadiffuser.core.utils.download_utils import check_download_integrity, get_repo_file_list, format_size

def check_download_status():
    """Check the current download status of FLUX.1-dev"""
    print("🔍 Checking FLUX.1-dev download status...\n")
    
    model_name = "flux.1-dev"
    model_path = settings.get_model_path(model_name)
    
    # Check if model is in registry
    if model_name not in model_manager.model_registry:
        print("❌ FLUX.1-dev not found in model registry")
        return False
    
    model_info = model_manager.model_registry[model_name]
    repo_id = model_info["repo_id"]
    
    print(f"📦 Model: {model_name}")
    print(f"🔗 Repository: {repo_id}")
    print(f"📁 Local path: {model_path}")
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
        missing_files = []
        incomplete_files = []
        
        for expected_file, expected_size in file_sizes.items():
            local_file_path = model_path / expected_file
            if not local_file_path.exists():
                missing_files.append(expected_file)
            elif local_file_path.stat().st_size != expected_size:
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
    
    # Check if download process is running
    print("🔍 Checking for active download processes...")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'ollamadiffuser pull flux.1-dev' in result.stdout:
            print("🔄 Download process is currently running")
            return "downloading"
        else:
            print("💤 No active download process found")
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
    
    return "incomplete"

def main():
    """Main function"""
    print("🧪 FLUX.1-dev Download Status Checker\n")
    
    status = check_download_status()
    
    print("\n" + "="*60)
    
    if status is True:
        print("🎉 FLUX.1-dev is ready to use!")
        print("\n💡 You can now run:")
        print("   ollamadiffuser run flux.1-dev")
        
    elif status == "needs_config":
        print("🔧 Download complete but needs configuration")
        print("\n💡 Run this to add to configuration:")
        print("   python -c \"from ollamadiffuser.core.models.manager import model_manager; model_manager.pull_model('flux.1-dev')\"")
        
    elif status == "downloading":
        print("⏳ Download is in progress")
        print("\n💡 Wait for download to complete, or check progress with:")
        print("   ps aux | grep 'ollamadiffuser pull'")
        
    elif status == "incomplete":
        print("⚠️ Download is incomplete")
        print("\n💡 Resume download with:")
        print("   ollamadiffuser pull flux.1-dev")
        print("\n💡 Or force fresh download with:")
        print("   ollamadiffuser pull flux.1-dev --force")
        
    else:
        print("❌ FLUX.1-dev not downloaded")
        print("\n💡 Start download with:")
        print("   ollamadiffuser pull flux.1-dev")
    
    print("\n📚 For more help, see: FLUX_USAGE_GUIDE.md")

if __name__ == "__main__":
    main() 