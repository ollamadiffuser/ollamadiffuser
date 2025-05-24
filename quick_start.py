#!/usr/bin/env python3
"""
OllamaDiffuser 快速开始脚本
自动安装依赖、测试环境，并提供快速入门指导
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def run_command(cmd, description, check=True):
    """运行命令并显示进度"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ {description} 完成")
            return True
        else:
            print(f"  ❌ {description} 失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ {description} 异常: {e}")
        return False

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} 版本合适")
        return True
    else:
        print(f"  ❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("     需要Python 3.8或更高版本")
        return False

def install_dependencies():
    """安装依赖"""
    print("\n📦 安装依赖包...")
    
    # 检查是否在虚拟环境中
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("  ⚠️  建议在虚拟环境中运行")
        print("     创建虚拟环境: python -m venv venv")
        print("     激活虚拟环境: source venv/bin/activate (Linux/Mac) 或 venv\\Scripts\\activate (Windows)")
        
        response = input("  是否继续在当前环境安装? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # 升级pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "升级pip"):
        return False
    
    # 安装requirements.txt
    if Path("requirements.txt").exists():
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "安装依赖包"):
            return False
    else:
        print("  ❌ requirements.txt文件不存在")
        return False
    
    # 安装项目本身
    if not run_command(f"{sys.executable} -m pip install -e .", "安装OllamaDiffuser"):
        return False
    
    return True

def test_installation():
    """测试安装"""
    print("\n🧪 测试安装...")
    
    try:
        # 测试导入
        import ollamadiffuser
        print("  ✅ OllamaDiffuser导入成功")
        
        # 运行安装测试
        if Path("test_installation.py").exists():
            result = subprocess.run([sys.executable, "test_installation.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("  ✅ 安装测试通过")
                return True
            else:
                print("  ❌ 安装测试失败")
                print(result.stdout)
                return False
        else:
            print("  ✅ 基本导入测试通过")
            return True
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False

def setup_environment():
    """设置环境变量"""
    print("\n⚙️  环境设置...")
    
    # 检查HuggingFace token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("  ⚠️  未设置HuggingFace token")
        print("     某些模型需要HuggingFace账户才能下载")
        print("     设置方法: export HF_TOKEN=your_token_here")
        
        token = input("  请输入您的HuggingFace token (可选): ").strip()
        if token:
            os.environ['HF_TOKEN'] = token
            print("  ✅ HuggingFace token已设置（临时）")
        else:
            print("  ⚠️  跳过HuggingFace token设置")
    else:
        print("  ✅ HuggingFace token已设置")

def show_quick_start_guide():
    """显示快速开始指南"""
    print("\n🚀 快速开始指南")
    print("=" * 50)
    
    print("\n1. 查看可用模型:")
    print("   ollamadiffuser list")
    
    print("\n2. 下载模型 (推荐从小模型开始):")
    print("   ollamadiffuser pull stable-diffusion-1.5")
    print("   # 或者更大的模型:")
    print("   ollamadiffuser pull stable-diffusion-3.5-medium")
    
    print("\n3. 运行模型服务:")
    print("   ollamadiffuser run stable-diffusion-1.5")
    
    print("\n4. 在另一个终端生成图像:")
    print('   curl -X POST http://localhost:8000/api/generate \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"prompt": "A beautiful sunset over mountains"}\' \\')
    print('     --output image.png')
    
    print("\n5. 或者启动Web UI:")
    print("   python -m ollamadiffuser --mode ui")
    print("   然后访问: http://localhost:8001")
    
    print("\n6. 其他有用的命令:")
    print("   ollamadiffuser show MODEL_NAME     # 查看模型信息")
    print("   ollamadiffuser ps                  # 查看运行状态")
    print("   ollamadiffuser unload              # 卸载模型")
    print("   ollamadiffuser rm MODEL_NAME       # 删除模型")
    
    print("\n📚 更多信息:")
    print("   - 项目文档: README.md")
    print("   - 项目结构: PROJECT_STRUCTURE.md")
    print("   - 演示脚本: python demo.py")

def main():
    """主函数"""
    print("🎨 OllamaDiffuser 快速开始")
    print("=" * 50)
    print("一个类似Ollama的图像生成模型管理工具")
    print()
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 安装依赖
    if not install_dependencies():
        print("\n❌ 依赖安装失败，请检查错误信息")
        sys.exit(1)
    
    # 测试安装
    if not test_installation():
        print("\n❌ 安装测试失败，请检查错误信息")
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 显示快速开始指南
    show_quick_start_guide()
    
    print("\n🎉 安装完成！")
    print("现在您可以开始使用OllamaDiffuser了!")

if __name__ == "__main__":
    main() 