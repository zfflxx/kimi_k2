### 来读一下deepseek moe的实现

- @moe.note.md @deepseek_v2_moe.tex @deepseek_v3_arch.tex
- @moe.note.md @deepseek_v3_arch.tex

- 帮我概览的总结一下deepseek的MOE原理
  - [DeepSeek MOE原理概览总结](./deepseek_moe_summary.md)
  - 帮我理解一下, 比如在训练中batch=N, N是一个很大的数字, 那么这个Batch会尽量平均的路由到每个设备上吗? 推理过程中也是一样, 那么如果token分布不均呢?
    - [MOE负载均衡机制分析](./moe_load_balancing_analysis.md)
  - V2中的token丢弃和V3中的动态负载监控都将本来计算出来的专家因为负载问题而切换, 这样会影响最终推理的效果吗? 影响多大或为什么不?
    - [MOE路由干预对推理效果的影响分析](./moe_routing_impact_analysis.md)
  - 详细解释一下V2与V3的路由机制差异? 为什么V3要做这个升级?
    - [V2与V3路由机制详细对比分析](./v2_v3_routing_comparison.md)

- 帮我理解Auxiliary-Loss-Free Load Balancing
  - [Auxiliary-Loss-Free Load Balancing详解](./auxiliary_loss_free_load_balancing.md)
  - 这个偏置b在训练阶段是如何被学习和在推理阶段是如何被使用的?
    - [偏置项训练与推理机制详解](./bias_training_inference.md)
  - 如果这个偏置和梯度下降是完全隔离的, 那么这个偏置的变更会影响梯度本来的计算, 会不会产生冲突?
    - [偏置与梯度冲突深度分析](./bias_gradient_conflict_analysis.md)

- 帮我理解Complementary Sequence-Wise Auxiliary Loss
  - [序列级辅助损失详解](./complementary_sequence_wise_auxiliary_loss.md)

- 帮我理解Node-Limited Routing
  - [Node-Limited Routing机制详解](./node_limited_routing_analysis.md)
  - 这个有放在训练目标中吗? 如果没放的话, 会不会路由选择总是会是次优的选择?
    - [Node-Limited Routing训练目标与路由优化分析](./node_routing_training_objective_analysis.md)