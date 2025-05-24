#!/usr/bin/env python3
"""
OllamaDiffuser 安装测试脚本
验证所有组件是否正确安装和配置
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_imports():
    """测试所有必要的包导入"""
    print("🔍 测试包导入...")
    
    required_packages = [
        'torch',
        'diffusers',
        'transformers',
        'accelerate',
        'litserve',
        'fastapi',
        'uvicorn',
        'huggingface_hub',
        'PIL',
        'click',
        'rich',
        'pydantic'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'huggingface_hub':
                importlib.import_module('huggingface_hub')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ 失败的导入: {', '.join(failed_imports)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有必要包导入成功!")
        return True

def test_ollamadiffuser_imports():
    """测试OllamaDiffuser组件导入"""
    print("\n🔍 测试OllamaDiffuser组件导入...")
    
    components = [
        'ollamadiffuser.core.config.settings',
        'ollamadiffuser.core.models.manager',
        'ollamadiffuser.core.inference.engine',
        'ollamadiffuser.api.server',
        'ollamadiffuser.cli.main',
        'ollamadiffuser.ui.web'
    ]
    
    failed_imports = []
    
    for component in components:
        try:
            importlib.import_module(component)
            print(f"  ✅ {component}")
        except ImportError as e:
            print(f"  ❌ {component}: {e}")
            failed_imports.append(component)
    
    if failed_imports:
        print(f"\n❌ 失败的组件导入: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ 所有OllamaDiffuser组件导入成功!")
        return True

def test_hardware():
    """测试硬件支持"""
    print("\n🔍 测试硬件支持...")
    
    try:
        import torch
        
        # 测试CUDA
        if torch.cuda.is_available():
            print(f"  ✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA 版本: {torch.version.cuda}")
            print(f"     GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  CUDA 不可用")
        
        # 测试MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon) 可用")
        else:
            print("  ⚠️  MPS 不可用")
        
        print(f"  ✅ CPU 可用: {torch.get_num_threads()} 线程")
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 硬件测试失败: {e}")
        return False

def test_cli():
    """测试CLI命令"""
    print("\n🔍 测试CLI命令...")
    
    try:
        # 测试help命令
        result = subprocess.run([
            sys.executable, '-m', 'ollamadiffuser.cli.main', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ CLI help 命令工作正常")
        else:
            print(f"  ❌ CLI help 命令失败: {result.stderr}")
            return False
        
        # 测试list命令
        result = subprocess.run([
            sys.executable, '-m', 'ollamadiffuser.cli.main', 'list'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ CLI list 命令工作正常")
        else:
            print(f"  ❌ CLI list 命令失败: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI测试失败: {e}")
        return False

def test_config():
    """测试配置系统"""
    print("\n🔍 测试配置系统...")
    
    try:
        from ollamadiffuser.core.config.settings import settings
        
        print(f"  ✅ 配置目录: {settings.config_dir}")
        print(f"  ✅ 模型目录: {settings.models_dir}")
        print(f"  ✅ 缓存目录: {settings.cache_dir}")
        print(f"  ✅ 服务器配置: {settings.server.host}:{settings.server.port}")
        
        # 检查目录是否创建
        if settings.config_dir.exists():
            print("  ✅ 配置目录已创建")
        else:
            print("  ❌ 配置目录未创建")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")
        return False

def test_model_manager():
    """测试模型管理器"""
    print("\n🔍 测试模型管理器...")
    
    try:
        from ollamadiffuser.core.models.manager import model_manager
        
        # 测试可用模型列表
        available_models = model_manager.list_available_models()
        print(f"  ✅ 可用模型: {len(available_models)} 个")
        for model in available_models:
            print(f"    • {model}")
        
        # 测试已安装模型列表
        installed_models = model_manager.list_installed_models()
        print(f"  ✅ 已安装模型: {len(installed_models)} 个")
        
        # 测试模型信息
        if available_models:
            model_name = available_models[0]
            info = model_manager.get_model_info(model_name)
            if info:
                print(f"  ✅ 模型信息获取成功: {model_name}")
            else:
                print(f"  ❌ 模型信息获取失败: {model_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 OllamaDiffuser 安装测试")
    print("=" * 50)
    
    tests = [
        ("依赖包导入", test_imports),
        ("组件导入", test_ollamadiffuser_imports),
        ("硬件支持", test_hardware),
        ("配置系统", test_config),
        ("模型管理器", test_model_manager),
        ("CLI命令", test_cli),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! OllamaDiffuser 安装成功!")
        print("\n🚀 快速开始:")
        print("1. 查看可用模型: ollamadiffuser list")
        print("2. 下载模型: ollamadiffuser pull stable-diffusion-3.5-medium")
        print("3. 运行服务: ollamadiffuser run stable-diffusion-3.5-medium")
        return True
    else:
        print("❌ 有测试失败，请检查安装!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 