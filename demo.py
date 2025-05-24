#!/usr/bin/env python3
"""
OllamaDiffuser 演示脚本
展示如何使用编程方式访问模型管理和图像生成功能
"""

import asyncio
import requests
import json
from pathlib import Path

# 导入OllamaDiffuser组件
from ollamadiffuser.core.models.manager import model_manager
from ollamadiffuser.core.config.settings import settings

def demo_model_management():
    """演示模型管理功能"""
    print("🔧 OllamaDiffuser 模型管理演示")
    print("=" * 50)
    
    # 列出可用模型
    print("\n📋 可用模型:")
    available_models = model_manager.list_available_models()
    for model in available_models:
        print(f"  • {model}")
    
    # 列出已安装模型
    print("\n✅ 已安装模型:")
    installed_models = model_manager.list_installed_models()
    if installed_models:
        for model in installed_models:
            print(f"  • {model}")
    else:
        print("  (无)")
    
    # 获取模型信息
    print("\n🔍 模型详细信息:")
    for model_name in available_models[:2]:  # 只展示前两个
        info = model_manager.get_model_info(model_name)
        if info:
            print(f"  {model_name}:")
            print(f"    类型: {info.get('model_type')}")
            print(f"    变体: {info.get('variant')}")
            print(f"    已安装: {info.get('installed', False)}")

def demo_api_client():
    """演示API客户端功能"""
    print("\n🌐 API客户端演示")
    print("=" * 50)
    
    base_url = f"http://{settings.server.host}:{settings.server.port}"
    
    try:
        # 检查健康状态
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API服务器连接成功")
            print(f"   模型已加载: {health_data.get('model_loaded', False)}")
            print(f"   当前模型: {health_data.get('current_model', 'None')}")
        else:
            print("❌ API服务器连接失败")
            return
        
        # 获取模型列表
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            models_data = response.json()
            print(f"\n📋 可用模型: {len(models_data.get('available', []))}")
            print(f"   已安装模型: {len(models_data.get('installed', []))}")
        
        # 如果有模型加载，尝试生成图像
        if health_data.get('model_loaded', False):
            print("\n🎨 尝试生成图像...")
            generate_data = {
                "prompt": "A beautiful sunset over mountains",
                "negative_prompt": "low quality, blurry",
                "num_inference_steps": 4,  # 使用较少步数以节省时间
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
                # 保存图像
                output_path = Path("demo_output.png")
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ 图像生成成功，已保存到: {output_path}")
            else:
                print(f"❌ 图像生成失败: {response.status_code}")
        else:
            print("⚠️  没有模型加载，跳过图像生成演示")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器")
        print("   请先启动服务器: ollamadiffuser serve")
    except Exception as e:
        print(f"❌ API演示出错: {e}")

def print_usage_examples():
    """打印使用示例"""
    print("\n📚 使用示例")
    print("=" * 50)
    
    examples = [
        ("查看所有可用模型", "ollamadiffuser list"),
        ("下载模型", "ollamadiffuser pull stable-diffusion-3.5-medium"),
        ("运行模型服务", "ollamadiffuser run stable-diffusion-3.5-medium"),
        ("启动API服务器", "ollamadiffuser serve"),
        ("启动Web UI", "python -m ollamadiffuser --mode ui"),
        ("查看模型信息", "ollamadiffuser show stable-diffusion-3.5-medium"),
        ("卸载模型", "ollamadiffuser unload"),
        ("删除模型", "ollamadiffuser rm stable-diffusion-3.5-medium"),
    ]
    
    for desc, cmd in examples:
        print(f"  {desc}:")
        print(f"    {cmd}")
        print()

def main():
    """主演示函数"""
    print("🎨 OllamaDiffuser 演示")
    print("一个类似 Ollama 的图像生成模型管理工具")
    print("=" * 60)
    
    # 显示配置信息
    print(f"\n⚙️  配置信息:")
    print(f"   配置目录: {settings.config_dir}")
    print(f"   模型目录: {settings.models_dir}")
    print(f"   服务器地址: {settings.server.host}:{settings.server.port}")
    
    # 演示模型管理
    demo_model_management()
    
    # 演示API客户端
    demo_api_client()
    
    # 打印使用示例
    print_usage_examples()
    
    print("🎉 演示完成！")
    print("\n快速开始:")
    print("1. 下载模型: ollamadiffuser pull stable-diffusion-3.5-medium")
    print("2. 运行服务: ollamadiffuser run stable-diffusion-3.5-medium")
    print("3. 访问 Web UI: python -m ollamadiffuser --mode ui")

if __name__ == "__main__":
    main() 