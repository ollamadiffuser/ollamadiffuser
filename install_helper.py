#!/usr/bin/env python3
"""
Installation Helper for OllamaDiffuser

This script helps users install OllamaDiffuser with the correct syntax for their shell.
"""

import os
import sys
import subprocess

def detect_shell():
    """Detect the current shell"""
    shell = os.environ.get('SHELL', '')
    if 'zsh' in shell:
        return 'zsh'
    elif 'fish' in shell:
        return 'fish'
    elif 'bash' in shell:
        return 'bash'
    elif 'sh' in shell:
        return 'sh'
    else:
        return 'unknown'

def get_install_command(shell, package='ollamadiffuser', extras='full'):
    """Get the correct install command for the shell"""
    base_cmd = f"pip install {package}"
    
    if extras:
        if shell in ['zsh']:
            return f'{base_cmd} "{package}[{extras}]"'
        elif shell in ['fish']:
            return f"{base_cmd} '{package}[{extras}]'"
        else:  # bash, sh, or unknown
            return f"{base_cmd} {package}[{extras}]"
    else:
        return base_cmd

def main():
    print("🎨 OllamaDiffuser Installation Helper")
    print("=" * 40)
    
    shell = detect_shell()
    print(f"Detected shell: {shell}")
    
    print("\n📦 Installation Commands:")
    print("-" * 25)
    
    # Basic installation
    basic_cmd = get_install_command(shell, 'ollamadiffuser', None)
    print(f"Basic:     {basic_cmd}")
    
    # Full installation with all optional dependencies
    full_cmd = get_install_command(shell, 'ollamadiffuser', 'full')
    print(f"Full:      {full_cmd}")
    
    # Development installation
    dev_cmd = get_install_command(shell, 'ollamadiffuser', 'dev')
    print(f"Dev:       {dev_cmd}")
    
    print("\n🔧 What each option includes:")
    print("• Basic: Core functionality (without ControlNet)")
    print("• Full:  Core + ControlNet + OpenCV (recommended)")
    print("• Dev:   Full + development tools (pytest, black, etc.)")
    
    print("\n💡 Recommendation:")
    print(f"For most users, run: {full_cmd}")
    
    # Offer to install
    print("\n❓ Would you like to install OllamaDiffuser now? (y/n)")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes']:
            print(f"\n🚀 Running: {full_cmd}")
            result = subprocess.run(full_cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Installation successful!")
                print("\n🎯 Next steps:")
                print("1. ollamadiffuser pull stable-diffusion-1.5")
                print("2. ollamadiffuser run stable-diffusion-1.5")
                print("3. Visit http://localhost:8001 for the web UI")
            else:
                print("❌ Installation failed:")
                print(result.stderr)
        else:
            print("Installation cancelled.")
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")

if __name__ == "__main__":
    main() 