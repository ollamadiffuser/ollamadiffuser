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
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """OllamaDiffuser - Image generation model management tool"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

@cli.command()
@click.argument('model_name')
@click.option('--force', '-f', is_flag=True, help='Force re-download')
def pull(model_name: str, force: bool):
    """Download model"""
    rprint(f"[blue]Downloading model: {model_name}[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Downloading {model_name}...", total=None)
        
        def progress_callback(message: str):
            """Update progress display with download status"""
            progress.update(task, description=message)
        
        if model_manager.pull_model(model_name, force=force, progress_callback=progress_callback):
            progress.update(task, description=f"✅ {model_name} download completed")
            rprint(f"[green]Model {model_name} downloaded successfully![/green]")
        else:
            progress.update(task, description=f"❌ {model_name} download failed")
            rprint(f"[red]Model {model_name} download failed![/red]")
            sys.exit(1)

@cli.command()
@click.argument('model_name')
@click.option('--host', '-h', default=None, help='Server host address')
@click.option('--port', '-p', default=None, type=int, help='Server port')
def run(model_name: str, host: Optional[str], port: Optional[int]):
    """Run model service"""
    rprint(f"[blue]Starting model service: {model_name}[/blue]")
    
    # Check if model is installed
    if not model_manager.is_model_installed(model_name):
        rprint(f"[red]Model {model_name} is not installed. Please run first: ollamadiffuser pull {model_name}[/red]")
        sys.exit(1)
    
    # Load model
    rprint("[yellow]Loading model...[/yellow]")
    if not model_manager.load_model(model_name):
        rprint(f"[red]Failed to load model {model_name}![/red]")
        sys.exit(1)
    
    rprint(f"[green]Model {model_name} loaded successfully![/green]")
    
    # Start server
    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        rprint("\n[yellow]Server stopped[/yellow]")
        model_manager.unload_model()
        # Clear the current model from settings when server stops
        settings.current_model = None
        settings.save_config()

@cli.command()
@click.option('--hardware', '-hw', is_flag=True, help='Show hardware requirements')
def list(hardware: bool):
    """List all models"""
    available_models = model_manager.list_available_models()
    installed_models = model_manager.list_installed_models()
    current_model = model_manager.get_current_model()
    
    if hardware:
        # Show detailed hardware requirements
        for model_name in available_models:
            info = model_manager.get_model_info(model_name)
            if not info:
                continue
                
            # Check installation status
            if model_name in installed_models:
                status = "✅ Installed"
                if model_name == current_model:
                    status += " (current)"
                size = info.get('size', 'Unknown')
            else:
                status = "⬇️ Available"
                size = "-"
            
            # Create individual table for each model
            table = Table(title=f"[bold cyan]{model_name}[/bold cyan] - {status}")
            table.add_column("Property", style="yellow", no_wrap=True)
            table.add_column("Value", style="white")
            
            # Basic info
            table.add_row("Type", info.get('model_type', 'Unknown'))
            table.add_row("Size", size)
            
            # Hardware requirements
            hw_req = info.get('hardware_requirements', {})
            if hw_req:
                table.add_row("Min VRAM", f"{hw_req.get('min_vram_gb', 'Unknown')} GB")
                table.add_row("Recommended VRAM", f"{hw_req.get('recommended_vram_gb', 'Unknown')} GB")
                table.add_row("Min RAM", f"{hw_req.get('min_ram_gb', 'Unknown')} GB")
                table.add_row("Recommended RAM", f"{hw_req.get('recommended_ram_gb', 'Unknown')} GB")
                table.add_row("Disk Space", f"{hw_req.get('disk_space_gb', 'Unknown')} GB")
                table.add_row("Supported Devices", ", ".join(hw_req.get('supported_devices', [])))
                table.add_row("Performance Notes", hw_req.get('performance_notes', 'N/A'))
            
            console.print(table)
            console.print()  # Add spacing between models
    else:
        # Show compact table
        table = Table(title="OllamaDiffuser Model List")
        table.add_column("Model Name", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Size", style="blue")
        table.add_column("Type", style="magenta")
        table.add_column("Min VRAM", style="yellow")
        
        for model_name in available_models:
            # Check installation status
            if model_name in installed_models:
                status = "✅ Installed"
                if model_name == current_model:
                    status += " (current)"
                
                # Get model information
                info = model_manager.get_model_info(model_name)
                size = info.get('size', 'Unknown') if info else 'Unknown'
                model_type = info.get('model_type', 'Unknown') if info else 'Unknown'
            else:
                status = "⬇️ Available"
                size = "-"
                # Get type from registry
                info = model_manager.get_model_info(model_name)
                model_type = info.get('model_type', 'Unknown') if info else 'Unknown'
            
            # Get hardware requirements
            hw_req = info.get('hardware_requirements', {}) if info else {}
            min_vram = f"{hw_req.get('min_vram_gb', '?')} GB" if hw_req else "Unknown"
            
            table.add_row(model_name, status, size, model_type, min_vram)
        
        console.print(table)
        console.print("\n[dim]💡 Use --hardware flag to see detailed hardware requirements[/dim]")

@cli.command()
@click.argument('model_name')
def show(model_name: str):
    """Show model detailed information"""
    info = model_manager.get_model_info(model_name)
    
    if info is None:
        rprint(f"[red]Model {model_name} does not exist[/red]")
        sys.exit(1)
    
    rprint(f"[bold cyan]Model Information: {model_name}[/bold cyan]")
    rprint(f"Type: {info.get('model_type', 'Unknown')}")
    rprint(f"Variant: {info.get('variant', 'Unknown')}")
    rprint(f"Installed: {'Yes' if info.get('installed', False) else 'No'}")
    
    if info.get('installed', False):
        rprint(f"Local Path: {info.get('local_path', 'Unknown')}")
        rprint(f"Size: {info.get('size', 'Unknown')}")
    
    # Hardware requirements
    if 'hardware_requirements' in info and info['hardware_requirements']:
        hw_req = info['hardware_requirements']
        rprint("\n[bold]Hardware Requirements:[/bold]")
        rprint(f"  Min VRAM: {hw_req.get('min_vram_gb', 'Unknown')} GB")
        rprint(f"  Recommended VRAM: {hw_req.get('recommended_vram_gb', 'Unknown')} GB")
        rprint(f"  Min RAM: {hw_req.get('min_ram_gb', 'Unknown')} GB")
        rprint(f"  Recommended RAM: {hw_req.get('recommended_ram_gb', 'Unknown')} GB")
        rprint(f"  Disk Space: {hw_req.get('disk_space_gb', 'Unknown')} GB")
        rprint(f"  Supported Devices: {', '.join(hw_req.get('supported_devices', []))}")
        if hw_req.get('performance_notes'):
            rprint(f"  Performance Notes: {hw_req.get('performance_notes')}")
    
    if 'parameters' in info and info['parameters']:
        rprint("\n[bold]Default Parameters:[/bold]")
        for key, value in info['parameters'].items():
            rprint(f"  {key}: {value}")
    
    if 'components' in info and info['components']:
        rprint("\n[bold]Components:[/bold]")
        for key, value in info['components'].items():
            rprint(f"  {key}: {value}")

@cli.command()
@click.argument('model_name')
@click.confirmation_option(prompt='Are you sure you want to delete this model?')
def rm(model_name: str):
    """Remove model"""
    if model_manager.remove_model(model_name):
        rprint(f"[green]Model {model_name} removed successfully![/green]")
    else:
        rprint(f"[red]Failed to remove model {model_name}![/red]")
        sys.exit(1)

@cli.command()
def ps():
    """Show currently running model"""
    current_model = model_manager.get_current_model()
    server_running = model_manager.is_server_running()
    
    if current_model:
        rprint(f"[green]Current model: {current_model}[/green]")
        
        # Check server status
        if server_running:
            rprint(f"[green]Server status: Running on {settings.server.host}:{settings.server.port}[/green]")
            
            # Try to get model info from the running server
            try:
                import requests
                response = requests.get(f"http://{settings.server.host}:{settings.server.port}/api/models/running", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('loaded'):
                        info = data.get('info', {})
                        rprint(f"Device: {info.get('device', 'Unknown')}")
                        rprint(f"Type: {info.get('type', 'Unknown')}")
                        rprint(f"Variant: {info.get('variant', 'Unknown')}")
                    else:
                        rprint("[yellow]Model loaded but not active in server[/yellow]")
            except:
                pass
        else:
            rprint("[yellow]Server status: Not running[/yellow]")
            rprint("[dim]Model is set as current but server is not active[/dim]")
            
        # Show model info from local config
        model_info = model_manager.get_model_info(current_model)
        if model_info:
            rprint(f"Model type: {model_info.get('model_type', 'Unknown')}")
            if model_info.get('installed'):
                rprint(f"Size: {model_info.get('size', 'Unknown')}")
    else:
        if server_running:
            rprint("[yellow]Server is running but no model is loaded[/yellow]")
            rprint(f"[green]Server status: Running on {settings.server.host}:{settings.server.port}[/green]")
        else:
            rprint("[yellow]No model is currently running[/yellow]")
            rprint("[dim]Use 'ollamadiffuser run <model>' to start a model[/dim]")

@cli.command()
@click.option('--host', '-h', default=None, help='Server host address')
@click.option('--port', '-p', default=None, type=int, help='Server port')
def serve(host: Optional[str], port: Optional[int]):
    """Start API server (without loading model)"""
    rprint("[blue]Starting OllamaDiffuser API server...[/blue]")
    
    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        rprint("\n[yellow]Server stopped[/yellow]")

@cli.command()
@click.argument('model_name')
def load(model_name: str):
    """Load model into memory"""
    rprint(f"[blue]Loading model: {model_name}[/blue]")
    
    if model_manager.load_model(model_name):
        rprint(f"[green]Model {model_name} loaded successfully![/green]")
    else:
        rprint(f"[red]Failed to load model {model_name}![/red]")
        sys.exit(1)

@cli.command()
def unload():
    """Unload current model"""
    if model_manager.is_model_loaded():
        current_model = model_manager.get_current_model()
        model_manager.unload_model()
        rprint(f"[green]Model {current_model} unloaded[/green]")
    else:
        rprint("[yellow]No model to unload[/yellow]")

@cli.command()
def stop():
    """Stop running server"""
    if not model_manager.is_server_running():
        rprint("[yellow]No server is currently running[/yellow]")
        return
    
    try:
        import requests
        import signal
        import psutil
        
        host = settings.server.host
        port = settings.server.port
        
        # Try graceful shutdown via API first
        try:
            response = requests.post(f"http://{host}:{port}/api/shutdown", timeout=5)
            if response.status_code == 200:
                rprint("[green]Server stopped gracefully[/green]")
                return
        except:
            pass
        
        # Fallback: Find and terminate the process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('uvicorn' in arg for arg in cmdline) and any(str(port) in arg for arg in cmdline):
                    proc.terminate()
                    proc.wait(timeout=10)
                    rprint(f"[green]Server process (PID: {proc.info['pid']}) stopped[/green]")
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
        
        rprint("[red]Could not find or stop the server process[/red]")
        
    except ImportError:
        rprint("[red]psutil package required for stop command. Install with: pip install psutil[/red]")
    except Exception as e:
        rprint(f"[red]Failed to stop server: {e}[/red]")

@cli.group()
def lora():
    """LoRA (Low-Rank Adaptation) management commands"""
    pass

@lora.command()
@click.argument('repo_id')
@click.option('--weight-name', '-w', help='Specific weight file name (e.g., lora.safetensors)')
@click.option('--alias', '-a', help='Local alias name for the LoRA')
def pull(repo_id: str, weight_name: Optional[str], alias: Optional[str]):
    """Download LoRA weights from Hugging Face Hub"""
    from ..core.utils.lora_manager import lora_manager
    
    rprint(f"[blue]Downloading LoRA: {repo_id}[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Downloading LoRA...", total=None)
        
        def progress_callback(message: str):
            progress.update(task, description=message)
        
        if lora_manager.pull_lora(repo_id, weight_name=weight_name, alias=alias, progress_callback=progress_callback):
            progress.update(task, description=f"✅ LoRA download completed")
            rprint(f"[green]LoRA {repo_id} downloaded successfully![/green]")
        else:
            progress.update(task, description=f"❌ LoRA download failed")
            rprint(f"[red]LoRA {repo_id} download failed![/red]")
            sys.exit(1)

@lora.command()
@click.argument('lora_name')
@click.option('--scale', '-s', default=1.0, type=float, help='LoRA scale/strength (default: 1.0)')
def load(lora_name: str, scale: float):
    """Load LoRA weights into the current model"""
    from ..core.utils.lora_manager import lora_manager
    
    rprint(f"[blue]Loading LoRA: {lora_name} (scale: {scale})[/blue]")
    
    if lora_manager.load_lora(lora_name, scale=scale):
        rprint(f"[green]LoRA {lora_name} loaded successfully![/green]")
    else:
        rprint(f"[red]Failed to load LoRA {lora_name}![/red]")
        sys.exit(1)

@lora.command()
def unload():
    """Unload current LoRA weights"""
    from ..core.utils.lora_manager import lora_manager
    
    rprint("[blue]Unloading LoRA weights...[/blue]")
    
    if lora_manager.unload_lora():
        rprint("[green]LoRA weights unloaded successfully![/green]")
    else:
        rprint("[red]Failed to unload LoRA weights![/red]")
        sys.exit(1)

@lora.command()
@click.argument('lora_name')
@click.confirmation_option(prompt='Are you sure you want to delete this LoRA?')
def rm(lora_name: str):
    """Remove LoRA weights"""
    from ..core.utils.lora_manager import lora_manager
    
    rprint(f"[blue]Removing LoRA: {lora_name}[/blue]")
    
    if lora_manager.remove_lora(lora_name):
        rprint(f"[green]LoRA {lora_name} removed successfully![/green]")
    else:
        rprint(f"[red]Failed to remove LoRA {lora_name}![/red]")
        sys.exit(1)

@lora.command()
def list():
    """List available and installed LoRA weights"""
    from ..core.utils.lora_manager import lora_manager
    
    installed_loras = lora_manager.list_installed_loras()
    current_lora = lora_manager.get_current_lora()
    
    if not installed_loras:
        rprint("[yellow]No LoRA weights installed.[/yellow]")
        rprint("\n[dim]💡 Use 'ollamadiffuser lora pull <repo_id>' to download LoRA weights[/dim]")
        return
    
    table = Table(title="Installed LoRA Weights")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Repository", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Size", style="yellow")
    
    for lora_name, lora_info in installed_loras.items():
        status = "🔄 Loaded" if lora_name == current_lora else "💾 Available"
        size = lora_info.get('size', 'Unknown')
        repo_id = lora_info.get('repo_id', 'Unknown')
        
        table.add_row(lora_name, repo_id, status, size)
    
    console.print(table)

@lora.command()
@click.argument('lora_name')
def show(lora_name: str):
    """Show detailed LoRA information"""
    from ..core.utils.lora_manager import lora_manager
    
    lora_info = lora_manager.get_lora_info(lora_name)
    
    if not lora_info:
        rprint(f"[red]LoRA {lora_name} not found.[/red]")
        sys.exit(1)
    
    rprint(f"[bold cyan]LoRA Information: {lora_name}[/bold cyan]")
    rprint(f"Repository: {lora_info.get('repo_id', 'Unknown')}")
    rprint(f"Weight File: {lora_info.get('weight_name', 'Unknown')}")
    rprint(f"Local Path: {lora_info.get('path', 'Unknown')}")
    rprint(f"Size: {lora_info.get('size', 'Unknown')}")
    rprint(f"Downloaded: {lora_info.get('downloaded_at', 'Unknown')}")
    
    if lora_info.get('description'):
        rprint(f"Description: {lora_info.get('description')}")

@cli.command()
def version():
    """Show version information"""
    rprint("[bold cyan]OllamaDiffuser v1.0.0[/bold cyan]")
    rprint("Image generation model management tool")

if __name__ == '__main__':
    cli() 