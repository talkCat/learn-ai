# learn-ai

AI 学习工作区，当前主线是：

1. 先用 Hugging Face Transformers 建立工程直觉
2. 再补 Transformer 核心概念和 attention 形状
3. 手写一个可训练、可采样的极简 GPT
4. 对照阅读 `huggingface/transformers` 关键源码
5. 做一个可保存实验记录的 generation lab

## 目录

- `resources/`：材料索引与阅读清单
- `notes/`：阶段笔记与复盘
- `examples/`：短小可运行实验脚本
- `notebooks/`：Jupyter 交互式学习材料
- `projects/mini-gpt/`：手写 decoder-only GPT
- `projects/hf-generation-lab/`：Hugging Face 生成参数实验工具

## 环境配置（uv）

先安装 `uv`，然后在项目根目录运行：

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

后续命令建议用 `uv run python ...` 执行，不需要手动激活虚拟环境。

这里指定 Python 3.12 是为了优先使用 uv 管理的完整 Python 运行时；Jupyter kernel 依赖 `sqlite3`，部分系统自带 Python 构建可能缺少这个模块。

如果你在 Linux、macOS、Colab 或带 GPU 的机器上学习，优先按 PyTorch 官方安装页选择对应的 `torch` 安装命令，再用 `uv pip install -r requirements.txt` 安装其余依赖。

快速验证：

```bash
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
uv run python examples/05_attention_shapes.py
uv run python examples/06_causal_mask_demo.py
```

启动 Jupyter 学习材料：

```bash
uv run python -m ipykernel install --user --name learn-ai --display-name "Python (learn-ai)"
uv run jupyter lab notebooks

To access the server, open this file in a browser:
        file:///home/dev/.local/share/jupyter/runtime/jpserver-3542498-open.html
    Or copy and paste one of these URLs:
        http://localhost:8889/lab?token=9bc91f3d8fb3cf9a3cf72a3c2d517a8e5f44a493cca1eec7
        http://127.0.0.1:8889/lab?token=9bc91f3d8fb3cf9a3cf72a3c2d517a8e5f44a493cca1eec7
```

如果 notebook 报 `ModuleNotFoundError`，先确认当前 kernel 选择的是 `Python (learn-ai)`。
