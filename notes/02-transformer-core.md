# 02 Transformer 核心概念

## 本阶段要回答的问题

- embedding 如何把 `[batch, seq]` 变成 `[batch, seq, dim]`？
- self-attention 中 Q/K/V 的 shape 如何变化？
- multi-head attention 为什么要拆成多个 head？
- causal mask 如何阻止当前位置看未来？
- residual、layer norm、MLP 在 block 里分别做什么？
- BERT、GPT、T5 的架构差异对应什么任务差异？

## 建议执行

```powershell
python examples/05_attention_shapes.py
python examples/06_causal_mask_demo.py
```

## 关键 shape

```text
tokens:       [batch, seq]
embeddings:  [batch, seq, dim]
q/k/v:       [batch, heads, seq, head_dim]
scores:      [batch, heads, seq, seq]
output:      [batch, seq, dim]
```
