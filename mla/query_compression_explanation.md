# Query低秩压缩的理解

## 问题
为什么要对queries进行低秩压缩，即使它不能减少KV cache？

## 回答

根据论文中的表述：
> "in order to reduce the activation memory during training, we also perform low-rank compression for the queries, even if it cannot reduce the KV cache"

### 核心原因：减少训练时的激活内存

Query的低秩压缩主要是为了**减少训练时的激活内存(activation memory)**，而不是为了减少推理时的KV cache。

### 具体分析

1. **训练vs推理的不同需求**：
   - **推理时**：只需要缓存KV，query是当前token生成的，不需要缓存
   - **训练时**：需要保存所有中间激活用于反向传播，包括所有token的query激活

2. **激活内存的压力**：
   - 训练时需要存储前向传播的所有中间结果
   - Query的激活占用大量内存：$\mathbf{q}_t \in \mathbb{R}^{d_h n_h}$
   - 通过低秩压缩：$\mathbf{c}_t^Q \in \mathbb{R}^{d_c'} \rightarrow \mathbf{q}_t^C \in \mathbb{R}^{d_h n_h}$
   - 只需要缓存压缩后的 $\mathbf{c}_t^Q$，大大节省内存

3. **数学表示**：
   $$
   \mathbf{c}_{t}^{Q} = W^{DQ} \mathbf{h}_{t}
   $$
   $$
   \mathbf{q}_{t}^{C} = W^{UQ} \mathbf{c}_{t}^{Q}
   $$
   其中 $d_c' \ll d_h n_h$

4. **内存节省效果**：
   - 原始query激活：$d_h \times n_h$ 维度
   - 压缩后激活：$d_c'$ 维度 (其中 $d_c' \ll d_h n_h$)
   - 在训练的反向传播中只需要存储小得多的压缩表示

### 总结

Query的低秩压缩是MLA设计中一个重要的工程优化，专门针对训练时的内存效率。虽然它不能像KV压缩那样直接减少推理时的cache需求，但它通过减少训练时的激活内存占用，使得大模型的训练变得更加可行和高效。