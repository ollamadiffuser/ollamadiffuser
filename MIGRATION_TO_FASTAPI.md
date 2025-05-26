# 迁移到 FastAPI

## 概述

OllamaDiffuser 已经完全迁移到使用 FastAPI 而不是 LitServe。这个变化简化了依赖关系，提高了兼容性，并提供了更好的开发体验。

## 主要变化

### 1. 依赖项变化

**移除的依赖:**
- `litserve>=0.2.0`

**保留的依赖:**
- `fastapi>=0.100.0`
- `uvicorn>=0.23.0`

### 2. 服务器架构变化

**之前 (LitServe):**
```python
class ImageGenerationAPI(ls.LitAPI):
    def setup(self, device):
        # 初始化逻辑
    
    def predict(self, params):
        # 预测逻辑
    
    def encode_response(self, image):
        # 响应编码
```

**现在 (FastAPI):**
```python
def create_app() -> FastAPI:
    app = FastAPI(...)
    
    @app.post("/api/generate")
    async def generate_image(request: GenerateRequest):
        # 直接在端点中处理逻辑
        
    return app
```

### 3. 启动方式变化

**之前:**
```python
api = ImageGenerationAPI()
server = ls.LitServer(api, accelerator="auto")
server.run(host=host, port=port)
```

**现在:**
```python
app = create_app()
uvicorn.run(app, host=host, port=port, log_level="info")
```

## 优势

1. **简化依赖**: 移除了 LitServe 依赖，减少了潜在的兼容性问题
2. **更好的文档**: FastAPI 自动生成 OpenAPI 文档
3. **更广泛的社区支持**: FastAPI 有更大的社区和更多的资源
4. **更灵活的部署**: 可以使用任何 ASGI 服务器（uvicorn, gunicorn, hypercorn 等）
5. **更好的类型支持**: FastAPI 的类型系统更加完善

## API 兼容性

所有现有的 API 端点保持不变：

- `GET /api/health` - 健康检查
- `GET /api/models` - 列出模型
- `GET /api/models/running` - 获取运行中的模型
- `GET /api/models/{model_name}` - 获取模型信息
- `POST /api/models/pull` - 下载模型
- `POST /api/models/load` - 加载模型
- `POST /api/models/unload` - 卸载模型
- `DELETE /api/models/{model_name}` - 删除模型
- `POST /api/generate` - 生成图像

## 测试

运行以下命令来测试新的 FastAPI 服务器：

```bash
python test_fastapi_server.py
```

## 启动服务器

```bash
# 方法 1: 直接运行模块
python -m ollamadiffuser.api.server

# 方法 2: 使用 CLI 命令
ollamadiffuser serve

# 方法 3: 使用 uvicorn 直接运行
uvicorn ollamadiffuser.api.server:create_app --host 0.0.0.0 --port 8000
```

## 开发

对于开发环境，你可以启用自动重载：

```bash
uvicorn ollamadiffuser.api.server:create_app --reload --host 0.0.0.0 --port 8000
```

## 文档

FastAPI 自动生成的 API 文档可以在以下地址访问：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## 迁移检查清单

- [x] 移除 LitServe 依赖
- [x] 重构服务器代码使用纯 FastAPI
- [x] 更新 requirements.txt
- [x] 更新 setup.py
- [x] 更新测试脚本
- [x] 创建测试验证脚本
- [x] 更新文档
- [x] 验证所有 API 端点正常工作

## 故障排除

如果遇到问题，请检查：

1. 确保所有依赖都已正确安装：`pip install -r requirements.txt`
2. 运行测试脚本：`python test_fastapi_server.py`
3. 检查端口是否被占用
4. 查看服务器日志获取详细错误信息 