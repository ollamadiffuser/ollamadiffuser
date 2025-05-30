#!/usr/bin/env python3
"""
Quick script to check version consistency across files
"""
import re
import os

def get_version_from_init():
    """Get version from __init__.py"""
    with open("ollamadiffuser/__init__.py", "r") as f:
        content = f.read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        return match.group(1) if match else None

def get_version_from_pyproject():
    """Get version from pyproject.toml if it exists"""
    if not os.path.exists("pyproject.toml"):
        return None
    with open("pyproject.toml", "r") as f:
        content = f.read()
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        return match.group(1) if match else None

def main():
    init_version = get_version_from_init()
    pyproject_version = get_version_from_pyproject()
    
    print("Version Check:")
    print(f"__init__.py: {init_version}")
    print(f"setup.py: Now reads from __init__.py ✅")
    
    if pyproject_version:
        print(f"pyproject.toml: {pyproject_version}")
        if pyproject_version != init_version:
            print("❌ pyproject.toml version doesn't match!")
            print(f"Update pyproject.toml version to: {init_version}")
        else:
            print("✅ Versions are synchronized!")
    else:
        print("pyproject.toml: Not found")
    
    print(f"\n🎯 Single source of truth: ollamadiffuser/__init__.py = {init_version}")

if __name__ == "__main__":
    main() 