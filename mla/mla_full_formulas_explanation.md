# MLA完整公式详细讲解

Multi-Head Latent Attention (MLA) 的完整计算公式体现了其核心设计思想：通过低秩压缩和解耦位置编码来减少KV缓存。下面逐步详细解释每个公式：

## 1. Query压缩阶段

### 1.1 Query潜在向量生成
$$\mathbf{c}_{t}^{Q} = W^{DQ} \mathbf{h}_{t}$$

- **作用**：将输入隐藏状态 $\mathbf{h}_{t} \in \mathbb{R}^{d}$ 压缩为低维潜在向量
- **维度变化**：$\mathbb{R}^{d} \rightarrow \mathbb{R}^{d_c'}$，其中 $d_c' \ll d_h n_h$
- **目的**：减少训练时的激活内存（虽然不能减少KV缓存）
- **矩阵**：$W^{DQ} \in \mathbb{R}^{d_c' \times d}$ 是下投影矩阵

### 1.2 压缩Query恢复
$$[mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} = W^{UQ} \mathbf{c}_{t}^{Q}$$

- **作用**：从压缩的潜在向量恢复出多头的压缩query
- **维度变化**：$\mathbb{R}^{d_c'} \rightarrow \mathbb{R}^{d_h n_h}$
- **分头**：恢复后的向量被切分为 $n_h$ 个头，每头维度为 $d_h$
- **矩阵**：$W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c'}$ 是上投影矩阵

### 1.3 RoPE Query生成
$$[\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} = \operatorname{RoPE}({W^{QR}} \mathbf{c}_{t}^{Q})$$

- **作用**：为了解耦RoPE，单独生成带位置编码的query部分
- **RoPE应用**：$\operatorname{RoPE}(\cdot)$ 对向量应用旋转位置编码
- **维度**：每个头的RoPE部分维度为 $d_h^R$
- **矩阵**：$W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}$

### 1.4 Query拼接
$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}]$$

- **作用**：将压缩部分和RoPE部分拼接形成完整的query
- **最终维度**：每个头的query维度为 $d_h + d_h^R$
- **设计理念**：分离内容表示和位置信息

## 2. Key-Value压缩阶段

### 2.1 KV联合压缩（核心创新）
$$\boxed{\color{blue} \mathbf{c}_{t}^{KV}} = W^{DKV} \mathbf{h}_{t}$$

- **核心思想**：Key和Value共享同一个压缩表示
- **缓存对象**：这是推理时需要缓存的主要内容（蓝色框标注）
- **压缩比**：从 $d$ 维压缩到 $d_c$ 维，其中 $d_c \ll d_h n_h$
- **效率**：相比传统MHA缓存 $2n_h d_h$ 维，这里只需缓存 $d_c$ 维

### 2.2 压缩Key恢复
$$[\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} = W^{UK} \mathbf{c}_{t}^{KV}$$

- **作用**：从联合压缩向量恢复多头key
- **共享基础**：与value共享同一个压缩表示 $\mathbf{c}_{t}^{KV}$
- **推理优化**：实际推理时可通过矩阵结合律避免显式计算

### 2.3 RoPE Key生成
$$\boxed{\color{blue}\mathbf{k}_{t}^{R}} = \operatorname{RoPE}({W^{KR}} \mathbf{h}_{t})$$

- **解耦设计**：RoPE key不经过压缩，直接从原始输入生成
- **共享机制**：所有头共享同一个RoPE key
- **缓存需求**：这也是需要缓存的内容（蓝色框标注）
- **维度**：$\mathbf{k}_{t}^{R} \in \mathbb{R}^{d_h^R}$

### 2.4 Key拼接
$$\mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}]$$

- **结构对应**：与query结构保持一致
- **共享RoPE**：注意第 $i$ 头的key中，RoPE部分 $\mathbf{k}_{t}^{R}$ 是所有头共享的

### 2.5 Value恢复
$$[\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} = W^{UV} \mathbf{c}_{t}^{KV}$$

- **共享压缩**：与key共享同一个压缩表示
- **无RoPE**：Value不需要位置编码，只有压缩部分

## 3. 注意力计算

### 3.1 注意力权重计算
$$\mathbf{o}_{t, i} = \sum_{j=1}^{t} \operatorname{Softmax}_j(\frac{\mathbf{q}_{t, i}^T \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}$$

- **维度调整**：分母使用 $\sqrt{d_{h} + d_{h}^{R}}$ 因为query和key都是拼接后的向量
- **Value使用**：只使用压缩部分的value $\mathbf{v}_{j, i}^{C}$（因为value不需要位置信息）
- **计算范围**：对所有前缀token $j=1,\ldots,t$ 进行注意力计算

### 3.2 输出投影
$$\mathbf{u}_{t} = W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}]$$

- **多头融合**：将所有头的输出拼接后进行线性变换
- **最终输出**：得到注意力层的最终输出

## 4. 推理优化

在推理阶段，MLA利用矩阵乘法的结合律进行重要优化：

- **$W^{UK}$ 吸收**：可以将 $W^{UK}$ 吸收到 $W^{UQ}$ 中
- **$W^{UV}$ 吸收**：可以将 $W^{UV}$ 吸收到 $W^{O}$ 中
- **避免重计算**：这样就不需要显式计算 $\mathbf{k}_{t}^{C}$ 和 $\mathbf{v}_{t}^{C}$

## 5. 缓存分析

总缓存需求：$(d_{c} + d_h^R)l$ 个元素

其中：
- $d_c$：KV联合压缩维度（DeepSeek-V2中设为 $4d_h$）
- $d_h^R$：RoPE部分维度（设为 $\frac{d_h}{2}$）
- $l$：层数

相比传统MHA的 $2n_h d_h l$，大幅减少了缓存需求。

## 6. 设计精髓

1. **联合压缩**：Key和Value共享压缩表示，最大化压缩效率
2. **解耦RoPE**：将位置编码与压缩机制分离，保持推理效率
3. **分离设计**：内容信息走压缩路径，位置信息走独立路径
4. **推理优化**：通过矩阵吸收避免中间计算，提升推理速度

这种设计在保持甚至提升性能的同时，显著减少了推理时的内存占用。