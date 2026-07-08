# HF Generation Lab

一个命令行小工具，用来观察 Hugging Face causal LM 的生成链路。

功能：

- 输入 prompt
- 选择模型，默认 `sshleifer/tiny-gpt2`
- 调整 `temperature`、`top_p`、`max_new_tokens`
- 输出生成文本
- 显示 tokenizer 切分结果
- 显示 top-k next-token 候选
- 保存实验记录到 `runs/`

## 运行

```bash
cd projects/hf-generation-lab
uv run python generation_lab.py --prompt "A software engineer learns transformers by"
```

保存的结果包括：

- `runs/<timestamp>.json`
- `runs/<timestamp>.md`

## 常用参数

```bash
uv run python generation_lab.py \
  --model sshleifer/tiny-gpt2 \
  --prompt "The model predicts" \
  --max-new-tokens 40 \
  --temperature 0.8 \
  --top-p 0.9 \
  --top-k 10
```
