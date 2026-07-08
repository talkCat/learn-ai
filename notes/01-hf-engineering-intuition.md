# 01 Hugging Face 工程直觉

## 本阶段要回答的问题

- prompt 如何变成 token ids？
- `attention_mask` 为什么重要？
- 模型 forward 输出的 `logits` 是什么？
- `generate()` 为什么能一个 token 一个 token 生成？
- `max_new_tokens`、`temperature`、`top_p`、`num_beams` 分别控制什么？

## 建议执行

```powershell
python examples/01_pipeline_tasks.py
python examples/02_tokenizer_inspect.py
python examples/03_generation_params.py
python examples/04_logits_next_token.py
```

## 复盘模板

- 我观察到的 tokenizer 行为：
- generation 参数改变后，输出有什么变化：
- logits top-k 候选让我理解了什么：
- 还没弄懂的问题：
