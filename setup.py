from setuptools import setup, find_packages
import sys
import os

# Read version from __init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), "ollamadiffuser", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies that must be installed
REQUIRED = [
    "torch>=2.1.0",
    "diffusers>=0.26.0",
    "transformers>=4.35.0",
    "accelerate>=0.25.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.23.0",
    "huggingface-hub>=0.16.0",
    "Pillow>=9.0.0",
    "click>=8.0.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "protobuf>=3.20.0",
    "sentencepiece>=0.1.99",
    "safetensors>=0.3.0",
    "python-multipart>=0.0.0",
    "psutil>=5.9.0",
    "jinja2>=3.0.0",
    "peft>=0.10.0",
    "numpy>=1.21.0",
    "controlnet-aux>=0.0.7",
    "opencv-python>=4.8.0",
    "stable-diffusion-cpp-python>=0.1.0",
    "gguf>=0.1.0",
]

setup(
    name="ollamadiffuser",
    version=get_version(),
    author="OllamaDiffuser Team",
    author_email="ollamadiffuser@gmail.com",
    description="🎨 Ollama-like image generation model management tool with LoRA support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ollamadiffuser/ollamadiffuser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.10",
    install_requires=REQUIRED,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ollamadiffuser=ollamadiffuser.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 