# 00 学习地图

## 10 个核心词

| 词 | 是什么 | 工程里怎么看见它 |
| --- | --- | --- |
| token | 模型处理文本的最小片段，不一定等于单词 | tokenizer 输出的字符串片段 |
| token id | token 在词表里的整数编号 | `input_ids` |
| embedding | 把 token id 映射成向量 | `model.get_input_embeddings()` |
| logits | 模型对每个词表 token 的未归一化分数 | `outputs.logits` |
| attention | 当前 token 对其他 token 的加权读取 | attention scores / weights |
| causal mask | 阻止 GPT 看见未来 token 的上三角 mask | decoder-only 训练和生成 |
| context window | 模型一次能看到的最大 token 数 | `model.config.max_position_embeddings` |
| temperature | 调整采样随机性的温度参数 | `generate(temperature=...)` |
| top_p | nucleus sampling，保留累计概率达到 p 的候选 | `generate(top_p=...)` |
| fine-tuning | 在已有模型上继续训练以适配任务/风格 | `Trainer`、LoRA、SFT |

## 当前学习顺序

1. 跑 Hugging Face：先理解模型调用链路。
2. 看 Transformer 核心结构：把工程现象映射到 attention、mask、logits。
3. 手写 mini GPT：自己实现训练与采样。
4. 读关键源码：对照工业实现和教学实现的差异。
5. 做 generation lab：把参数、tokenizer、logits 可观察化。
