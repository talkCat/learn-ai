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
- `projects/mini-gpt/`：手写 decoder-only GPT
- `projects/hf-generation-lab/`：Hugging Face 生成参数实验工具

## 可选环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果你在 Linux、macOS、Colab 或带 GPU 的机器上学习，优先按 PyTorch 官方安装页选择对应命令，再安装其余依赖。

快速验证：

```powershell
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
python examples/05_attention_shapes.py
python examples/06_causal_mask_demo.py
```
