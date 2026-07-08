# Mini GPT

一个字符级 decoder-only Transformer，用来学习 GPT 的主干结构：

- token embedding
- position embedding
- causal self-attention
- MLP / feed-forward
- residual connection + layer norm
- next-token prediction
- autoregressive sampling

## 安装依赖

在仓库根目录安装：

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## 训练

```bash
cd projects/mini-gpt
uv run python train.py --max-iters 200 --eval-interval 50
```

CPU 上可以先用默认小配置跑通。如果有 GPU，可以增加 `--n-layer`、`--n-head`、`--n-embd`。

## 采样

```bash
uv run python sample.py --checkpoint checkpoints/mini_gpt.pt --prompt "To be"
```

## 学习重点

`data.py` 负责把文本变成整数序列；`model.py` 是 Transformer 主体；`train.py` 做 next-token prediction；`sample.py` 展示生成时如何把前文不断喂回模型。
