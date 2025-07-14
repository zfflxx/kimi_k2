### 读Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts

- @auxiliary-loss-free.tex @auxiliary-loss-free.note.md

- 帮我概览的总结一下这篇论文
  - [论文总结](auxiliary-loss-free-summary.md)

- 帮我解释这张图oss-Free Balancing selects experts according to a "biased gating score" in each training step and updates this expert-wise bias after each training step.
  - [算法流程图解释](loss-free-balancing-figure-explanation.md)
- 让我再理解一下MoE的路由, 在prefill阶段, 比如100个seq batch, 每个seq的tokens为1000, 那么这些每个seq的每个token都会被路由到不同的专家吗?
  - [MoE路由机制在Prefill阶段的详细解释](moe-routing-prefill-explanation.md)
- 如何理解$ \mathbf{e}_i$ is the centroid of the $i$-th expert
  - [专家的Centroid向量 $\mathbf{e}_i$ 的解释](expert-centroid-explanation.md)
  - 这里的输入$\mathbf{u}_t$只是token的embedding吗?还是这个token+这个token之前所有的上下文?
    - [输入$\mathbf{u}_t$的本质：Token Embedding还是包含上下文？](input-ut-context-explanation.md)
- 先帮我理解一下Auxiliary Loss
  - [Auxiliary Loss（辅助损失）详解](auxiliary-loss-explanation.md)
- MaxVio是一个什么样的指标? 如果不是这篇论文的实验结论, 为啥感觉通过梯度下降学习到的负载均衡比通过机器负载这种规则产生bias的负载均衡要好?
  - [MaxVio指标与梯度下降vs规则化bias的对比分析](maxvio-gradient-vs-bias-explanation.md)
  - 如何理解这是一个约束优化问题，但用无约束方法求解/这是真正的约束优化，数学上更合理
    - [约束优化问题的数学本质分析](constrained-optimization-explanation.md)

- 帮我理解小节The Dilemma Between Load Balance and Model Performance
  - [负载均衡与模型性能之间的矛盾解析](load-balance-performance-dilemma-explanation.md)
- 帮我理解小节Auxiliary-Loss-Free Load Balancing Strategy
  - [Auxiliary-Loss-Free Load Balancing Strategy 详解](auxiliary-loss-free-strategy-explanation.md)
  - 这里学习到的b_i并不是通过梯度下降学习到的, 而是根据负载和路由层输出计算出来的, 那么这些b_i参数在真正推理的时候是如何使用的呢?
    - [偏置参数 $b_i$ 在推理时的使用方式](bias-inference-usage-explanation.md)
  - 去sglang的代码去看看这个bias是如何利用的
  - 我本地的代码仓库有sglang/python/sglang/srt/layers/moe/topk.py, 帮我看看这个bias是如何利用的, 可以查看这个仓库的其他代码
    - [SGLang中correction_bias的使用分析](correction-bias-sglang-analysis.md)
  - 在训练的时候这些负载均衡学习的非常的好, 训练的语料是广泛分布的通用的互联网文本, 但是推理的时候, 假设主要的用户都用来将其用于生成代码的功能, 比如这是少量的专家, 那么是不是又出现了严重的负载不均衡?
    - [训练与推理分布偏移对负载均衡的影响分析](training-inference-distribution-shift-analysis.md)
  - 在推理阶段能否做到让负载重的那些专家比闲的expert多部署一些实例呢, 这样是不是就可以让负载均衡的更好?
    - [动态专家扩缩容：基于负载的专家实例部署策略](dynamic-expert-scaling-analysis.md)

- 给我详细深入解释实验章节
  - [实验章节详细解释](experiments-section-explanation.md)
- 给我深入解释讨论章节
  - [讨论章节详细解释](discussion-section-explanation.md)
- 帮我深入理解Future Token Leakage in Expert Choice
  - [Expert Choice中未来Token泄露的深入理解](future-token-leakage-expert-choice-explanation.md)
- 帮我深入理解Loss-Free Balancing Is Compatible with Expert Parallelism
  - [Loss-Free负载均衡与专家并行兼容性的深入理解](loss-free-expert-parallelism-compatibility.md)