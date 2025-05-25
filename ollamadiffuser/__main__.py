#!/usr/bin/env python3
import sys
import argparse
from .cli.main import cli
from .api.server import run_server
from .ui.web import create_ui_app
from .core.config.settings import settings

def main():
    """Main entry function"""
    parser = argparse.ArgumentParser(
        description='OllamaDiffuser - Image generation model management tool'
    )
    parser.add_argument(
        '--mode', 
        choices=['cli', 'api', 'ui'], 
        default='cli',
        help='Running mode: cli (command line), api (API server), ui (Web interface)'
    )
    parser.add_argument('--host', default=None, help='Server host address')
    parser.add_argument('--port', type=int, default=None, help='Server port')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args, unknown = parser.parse_known_args()
    
    if args.mode == 'cli':
        # Command line mode
        sys.argv = [sys.argv[0]] + unknown
        cli()
    elif args.mode == 'api':
        # API server mode
        print("Starting OllamaDiffuser API server...")
        run_server(host=args.host, port=args.port)
    elif args.mode == 'ui':
        # Web UI mode
        print("Starting OllamaDiffuser Web UI...")
        import uvicorn
        app = create_ui_app()
        host = args.host or settings.server.host
        port = args.port or (settings.server.port + 1)  # Web UI uses different port
        uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main() 