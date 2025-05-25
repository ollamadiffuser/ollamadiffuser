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
        
        if model_manager.pull_model(model_name, force=force):
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

@cli.command()
def list():
    """List all models"""
    available_models = model_manager.list_available_models()
    installed_models = model_manager.list_installed_models()
    current_model = model_manager.get_current_model()
    
    # Create table
    table = Table(title="OllamaDiffuser Model List")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Size", style="blue")
    table.add_column("Type", style="magenta")
    
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
            registry_info = model_manager.model_registry.get(model_name, {})
            model_type = registry_info.get('model_type', 'Unknown')
        
        table.add_row(model_name, status, size, model_type)
    
    console.print(table)

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
    if model_manager.is_model_loaded():
        current_model = model_manager.get_current_model()
        engine = model_manager.loaded_model
        info = engine.get_model_info() if engine else {}
        
        rprint(f"[green]Currently running model: {current_model}[/green]")
        if info:
            rprint(f"Device: {info.get('device', 'Unknown')}")
            rprint(f"Type: {info.get('type', 'Unknown')}")
            rprint(f"Variant: {info.get('variant', 'Unknown')}")
    else:
        rprint("[yellow]No model is currently running[/yellow]")

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
def version():
    """Show version information"""
    rprint("[bold cyan]OllamaDiffuser v1.0.0[/bold cyan]")
    rprint("Image generation model management tool")

if __name__ == '__main__':
    cli() 