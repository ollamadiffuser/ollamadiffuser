#!/usr/bin/env python3
import click
import sys
import logging
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from ..core.models.manager import model_manager
from ..core.config.settings import settings
from ..api.server import run_server

console = Console()

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='启用详细输出')
def cli(verbose):
    """OllamaDiffuser - 图像生成模型管理工具"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

@cli.command()
@click.argument('model_name')
@click.option('--force', '-f', is_flag=True, help='强制重新下载')
def pull(model_name: str, force: bool):
    """下载模型"""
    rprint(f"[blue]正在下载模型: {model_name}[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"下载 {model_name}...", total=None)
        
        if model_manager.pull_model(model_name, force=force):
            progress.update(task, description=f"✅ {model_name} 下载完成")
            rprint(f"[green]模型 {model_name} 下载成功![/green]")
        else:
            progress.update(task, description=f"❌ {model_name} 下载失败")
            rprint(f"[red]模型 {model_name} 下载失败![/red]")
            sys.exit(1)

@cli.command()
@click.argument('model_name')
@click.option('--host', '-h', default=None, help='服务器主机地址')
@click.option('--port', '-p', default=None, type=int, help='服务器端口')
def run(model_name: str, host: Optional[str], port: Optional[int]):
    """运行模型服务"""
    rprint(f"[blue]启动模型服务: {model_name}[/blue]")
    
    # 检查模型是否已安装
    if not model_manager.is_model_installed(model_name):
        rprint(f"[red]模型 {model_name} 未安装。请先运行: ollamadiffuser pull {model_name}[/red]")
        sys.exit(1)
    
    # 加载模型
    rprint("[yellow]正在加载模型...[/yellow]")
    if not model_manager.load_model(model_name):
        rprint(f"[red]加载模型 {model_name} 失败![/red]")
        sys.exit(1)
    
    rprint(f"[green]模型 {model_name} 加载成功![/green]")
    
    # 启动服务器
    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        rprint("\n[yellow]服务器已停止[/yellow]")
        model_manager.unload_model()

@cli.command()
def list():
    """列出所有模型"""
    available_models = model_manager.list_available_models()
    installed_models = model_manager.list_installed_models()
    current_model = model_manager.get_current_model()
    
    # 创建表格
    table = Table(title="OllamaDiffuser 模型列表")
    table.add_column("模型名称", style="cyan", no_wrap=True)
    table.add_column("状态", style="green")
    table.add_column("大小", style="blue")
    table.add_column("类型", style="magenta")
    
    for model_name in available_models:
        # 检查安装状态
        if model_name in installed_models:
            status = "✅ 已安装"
            if model_name == current_model:
                status += " (当前)"
            
            # 获取模型信息
            info = model_manager.get_model_info(model_name)
            size = info.get('size', 'Unknown') if info else 'Unknown'
            model_type = info.get('model_type', 'Unknown') if info else 'Unknown'
        else:
            status = "⬇️ 可下载"
            size = "-"
            # 从注册表获取类型
            registry_info = model_manager.model_registry.get(model_name, {})
            model_type = registry_info.get('model_type', 'Unknown')
        
        table.add_row(model_name, status, size, model_type)
    
    console.print(table)

@cli.command()
@click.argument('model_name')
def show(model_name: str):
    """显示模型详细信息"""
    info = model_manager.get_model_info(model_name)
    
    if info is None:
        rprint(f"[red]模型 {model_name} 不存在[/red]")
        sys.exit(1)
    
    rprint(f"[bold cyan]模型信息: {model_name}[/bold cyan]")
    rprint(f"类型: {info.get('model_type', 'Unknown')}")
    rprint(f"变体: {info.get('variant', 'Unknown')}")
    rprint(f"已安装: {'是' if info.get('installed', False) else '否'}")
    
    if info.get('installed', False):
        rprint(f"本地路径: {info.get('local_path', 'Unknown')}")
        rprint(f"大小: {info.get('size', 'Unknown')}")
    
    if 'parameters' in info and info['parameters']:
        rprint("\n[bold]默认参数:[/bold]")
        for key, value in info['parameters'].items():
            rprint(f"  {key}: {value}")
    
    if 'components' in info and info['components']:
        rprint("\n[bold]组件:[/bold]")
        for key, value in info['components'].items():
            rprint(f"  {key}: {value}")

@cli.command()
@click.argument('model_name')
@click.confirmation_option(prompt='确定要删除这个模型吗？')
def rm(model_name: str):
    """删除模型"""
    if model_manager.remove_model(model_name):
        rprint(f"[green]模型 {model_name} 删除成功![/green]")
    else:
        rprint(f"[red]删除模型 {model_name} 失败![/red]")
        sys.exit(1)

@cli.command()
def ps():
    """显示当前运行的模型"""
    if model_manager.is_model_loaded():
        current_model = model_manager.get_current_model()
        engine = model_manager.loaded_model
        info = engine.get_model_info() if engine else {}
        
        rprint(f"[green]当前运行的模型: {current_model}[/green]")
        if info:
            rprint(f"设备: {info.get('device', 'Unknown')}")
            rprint(f"类型: {info.get('type', 'Unknown')}")
            rprint(f"变体: {info.get('variant', 'Unknown')}")
    else:
        rprint("[yellow]没有模型正在运行[/yellow]")

@cli.command()
@click.option('--host', '-h', default=None, help='服务器主机地址')
@click.option('--port', '-p', default=None, type=int, help='服务器端口')
def serve(host: Optional[str], port: Optional[int]):
    """启动API服务器（不加载模型）"""
    rprint("[blue]启动OllamaDiffuser API服务器...[/blue]")
    
    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        rprint("\n[yellow]服务器已停止[/yellow]")

@cli.command()
@click.argument('model_name')
def load(model_name: str):
    """加载模型到内存"""
    rprint(f"[blue]正在加载模型: {model_name}[/blue]")
    
    if model_manager.load_model(model_name):
        rprint(f"[green]模型 {model_name} 加载成功![/green]")
    else:
        rprint(f"[red]加载模型 {model_name} 失败![/red]")
        sys.exit(1)

@cli.command()
def unload():
    """卸载当前模型"""
    if model_manager.is_model_loaded():
        current_model = model_manager.get_current_model()
        model_manager.unload_model()
        rprint(f"[green]模型 {current_model} 已卸载[/green]")
    else:
        rprint("[yellow]没有模型需要卸载[/yellow]")

@cli.command()
def version():
    """显示版本信息"""
    rprint("[bold cyan]OllamaDiffuser v1.0.0[/bold cyan]")
    rprint("图像生成模型管理工具")

if __name__ == '__main__':
    cli() 