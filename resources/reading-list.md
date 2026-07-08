# Transformer / Hugging Face / 手写 GPT 阅读清单

## P0：主线材料

| 材料 | 链接 | 用途 |
| --- | --- | --- |
| Hugging Face LLM Course | https://huggingface.co/learn/llm-course/chapter1/1 | 官方入门主线，覆盖 Transformers、Datasets、Tokenizers、微调 |
| huggingface/course | https://github.com/huggingface/course | HF Course 源码、章节内容和 notebook |
| Hugging Face Transformers Docs | https://huggingface.co/docs/transformers/index | API、pipeline、generate、Auto 类文档 |
| huggingface/transformers | https://github.com/huggingface/transformers | 工业级 Transformers 主库，Phase 4 源码阅读目标 |
| rasbt/LLMs-from-scratch | https://github.com/rasbt/LLMs-from-scratch | 从零实现 GPT/LLM 的系统教程 |
| karpathy/minGPT | https://github.com/karpathy/minGPT | 教育取向的极简 GPT 代码结构参考 |

## P1：补充材料

| 材料 | 链接 | 用途 |
| --- | --- | --- |
| karpathy/nanoGPT | https://github.com/karpathy/nanoGPT | 训练/微调 GPT 的简洁工程实现 |
| karpathy/nanochat | https://github.com/karpathy/nanochat | nanoGPT 之后更完整的端到端训练参考 |
| harvardnlp/annotated-transformer | https://github.com/harvardnlp/annotated-transformer | Transformer 论文实现和注释 |
| jaymody/picoGPT | https://github.com/jaymody/picoGPT | 极小 NumPy GPT-2 推理实现 |
| karpathy/makemore | https://github.com/karpathy/makemore | 字符级语言模型热身 |
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ | 可视化理解 attention 和 encoder/decoder |

## 使用方式

- Phase 1 先看 HF Course 与 Transformers Docs，配合 `examples/01_*` 到 `04_*`。
- Phase 2 看 `annotated-transformer` 和可视化文章，配合 `examples/05_*`、`06_*`。
- Phase 3 以 `projects/mini-gpt/` 为主，必要时对照 `LLMs-from-scratch` 和 `minGPT`。
- Phase 4 不读完整 `transformers` 仓库，只读 Auto 类、GPT-2/BERT modeling 文件、`generation` 主流程和 `Trainer` 概念。
