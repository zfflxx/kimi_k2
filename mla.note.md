### 来读一下MLA的论文

- 从DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model中将MLA的部分摘录出来
  - https://arxiv.org/abs/2405.04434
- 见[mla.tex](mla/mla.tex)
- @mla.note.md @mla.tex 

- 缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA
  - https://kexue.fm/archives/10091

- 帮我概览总结一下MLA的设计
  - [MLA设计概览总结](mla/mla_design_overview.md)
- 帮我详细解释Low-Rank Key-Value Joint Compression
  - [Low-Rank Key-Value Joint Compression详细解释](mla/low_rank_kv_compression.md)
- 如何理解in order to reduce the activation memory during training, we also perform low-rank compression for the queries, even if it cannot reduce the KV cache?
  - [Query低秩压缩的理解](mla/query_compression_explanation.md)
- 帮我详细解释Decoupled Rotary Position Embedding小节
  - [Decoupled RoPE详细解释](mla/decoupled_rope_explanation.md)
- 帮我详细理解Comparison of Key-Value Cache
  - [KV Cache比较详细分析](mla/kv_cache_comparison.md)
- 帮我详细的讲解公式 Full Formulas of MLA
  - [MLA完整公式详细讲解](mla/mla_full_formulas_explanation.md)
- 训练阶段跟Decoding阶段的Q,K并不一样, 详细解释一下
  - [训练与推理阶段Q,K差异详解](mla/training_vs_decoding_qk_difference.md)