# OllamaDiffuser 项目结构

```
ollamadiffuser/
├── README.md                    # 项目说明文档
├── setup.py                     # 安装配置文件
├── requirements.txt             # 依赖包列表
├── requirements-dev.txt         # 开发依赖包列表
├── demo.py                      # 演示脚本
├── test_installation.py         # 安装测试脚本
├── server_legacy.py            # 原始服务器代码（备份）
│
├── ollamadiffuser/             # 主包目录
│   ├── __init__.py             # 包初始化文件
│   ├── __main__.py             # 主入口文件
│   │
│   ├── core/                   # 核心功能模块
│   │   ├── __init__.py
│   │   ├── config/             # 配置管理
│   │   │   ├── __init__.py
│   │   │   └── settings.py     # 全局设置和配置管理
│   │   ├── models/             # 模型管理
│   │   │   ├── __init__.py
│   │   │   └── manager.py      # 模型下载、安装、管理器
│   │   └── inference/          # 推理引擎
│   │       ├── __init__.py
│   │       └── engine.py       # 图像生成推理引擎
│   │
│   ├── api/                    # API 服务器
│   │   ├── __init__.py
│   │   └── server.py           # FastAPI/LitServe 服务器
│   │
│   ├── cli/                    # 命令行接口
│   │   ├── __init__.py
│   │   └── main.py             # Click 命令行工具
│   │
│   ├── ui/                     # Web 用户界面
│   │   ├── __init__.py
│   │   ├── web.py              # FastAPI Web UI 应用
│   │   └── templates/          # HTML 模板
│   │       └── index.html      # 主页模板
│   │
│   └── utils/                  # 工具函数
│       └── __init__.py
│
└── ~/.ollamadiffuser/          # 用户配置目录 (运行时创建)
    ├── config.json             # 用户配置文件
    ├── models/                 # 模型存储目录
    │   ├── stable-diffusion-3.5-medium/
    │   ├── stable-diffusion-xl-base/
    │   └── ...
    └── cache/                  # 下载缓存目录
```

## 核心组件说明

### 1. 配置管理 (`core/config/`)
- **settings.py**: 全局设置管理，包括模型配置、服务器配置等
- 自动创建用户配置目录
- 支持 JSON 配置文件持久化

### 2. 模型管理 (`core/models/`)
- **manager.py**: 模型生命周期管理
  - 模型注册和发现
  - 下载和安装
  - 加载和卸载
  - 版本控制

### 3. 推理引擎 (`core/inference/`)
- **engine.py**: 图像生成推理
  - 支持多种 Stable Diffusion 变体
  - 设备自适应 (CUDA/MPS/CPU)
  - 内存优化
  - Prompt 处理

### 4. API 服务器 (`api/`)
- **server.py**: RESTful API 服务
  - 模型管理端点
  - 图像生成端点
  - 健康检查
  - CORS 支持

### 5. 命令行接口 (`cli/`)
- **main.py**: Ollama 风格的 CLI 工具
  - `pull` - 下载模型
  - `run` - 运行模型服务
  - `list` - 列出模型
  - `show` - 显示模型信息
  - `rm` - 删除模型
  - `ps` - 显示运行状态

### 6. Web 界面 (`ui/`)
- **web.py**: Web UI 应用
- **templates/**: HTML 模板
  - 图像生成界面
  - 模型状态显示
  - 参数调整

## 运行模式

### 1. CLI 模式
```bash
ollamadiffuser --mode cli [命令]
# 或直接使用
ollamadiffuser [命令]
```

### 2. API 服务器模式
```bash
ollamadiffuser --mode api
# 或
ollamadiffuser serve
```

### 3. Web UI 模式
```bash
ollamadiffuser --mode ui
```

## 数据流

1. **模型下载**: CLI → 模型管理器 → HuggingFace Hub
2. **模型加载**: 模型管理器 → 推理引擎 → GPU/CPU
3. **图像生成**: API/UI → 推理引擎 → Diffusers → PIL Image
4. **配置管理**: 所有组件 → 设置管理器 → JSON 文件

## 扩展点

1. **新模型支持**: 在 `manager.py` 中添加模型注册
2. **新推理后端**: 在 `engine.py` 中添加新的 pipeline 类
3. **新 API 端点**: 在 `server.py` 中添加新路由
4. **新 CLI 命令**: 在 `main.py` 中添加新的 Click 命令
5. **新 UI 组件**: 在 `ui/` 中添加新模板和路由 