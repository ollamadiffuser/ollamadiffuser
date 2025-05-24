#!/usr/bin/env python3
import sys
import argparse
from .cli.main import cli
from .api.server import run_server
from .ui.web import create_ui_app
from .core.config.settings import settings

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description='OllamaDiffuser - 图像生成模型管理工具'
    )
    parser.add_argument(
        '--mode', 
        choices=['cli', 'api', 'ui'], 
        default='cli',
        help='运行模式: cli (命令行), api (API服务器), ui (Web界面)'
    )
    parser.add_argument('--host', default=None, help='服务器主机地址')
    parser.add_argument('--port', type=int, default=None, help='服务器端口')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args, unknown = parser.parse_known_args()
    
    if args.mode == 'cli':
        # 命令行模式
        sys.argv = [sys.argv[0]] + unknown
        cli()
    elif args.mode == 'api':
        # API服务器模式
        print("启动OllamaDiffuser API服务器...")
        run_server(host=args.host, port=args.port)
    elif args.mode == 'ui':
        # Web UI模式
        print("启动OllamaDiffuser Web UI...")
        import uvicorn
        app = create_ui_app()
        host = args.host or settings.server.host
        port = args.port or (settings.server.port + 1)  # Web UI使用不同端口
        uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main() 