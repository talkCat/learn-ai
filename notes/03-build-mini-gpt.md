# 03 手写 Mini GPT

## 本阶段要回答的问题

- 为什么语言模型训练目标是用 `x` 预测右移一位的 `y`？
- causal self-attention 和普通 self-attention 的差异在哪里？
- loss 下降说明了什么，不说明什么？
- 采样时 temperature 和 top-k 如何影响下一个字符？

## 建议执行

```powershell
cd projects/mini-gpt
python train.py --max-iters 200 --eval-interval 50
python sample.py --checkpoint checkpoints/mini_gpt.pt --prompt "To be"
```

## 复盘模板

- 第一次跑通训练的 loss：
- 生成文本里最明显的模式：
- 我最容易搞错的 shape：
- 如果要扩展到 BPE tokenizer，需要改哪里：
