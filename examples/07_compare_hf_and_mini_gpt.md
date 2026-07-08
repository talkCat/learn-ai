# HF GPT-2 与 Mini GPT 对照

| 概念 | Hugging Face GPT-2 | `projects/mini-gpt` |
| --- | --- | --- |
| 配置 | `GPT2Config` / `AutoConfig` | `GPTConfig` |
| token embedding | `wte` | `token_embedding_table` |
| position embedding | `wpe` | `position_embedding_table` |
| block | `GPT2Block` | `Block` |
| causal attention | `GPT2Attention` | `CausalSelfAttention` |
| MLP | `GPT2MLP` | `FeedForward` |
| forward 输出 | logits、hidden states、attentions 等 | logits、loss |
| 生成 | `generate()` 通用框架 | `GPT.generate()` 简化采样 |

## 阅读提示

工业库多出的复杂度通常来自配置、权重兼容、KV cache、设备管理、量化、分布式、导出和历史模型兼容。教学实现只保留最短路径，适合理解主干。
