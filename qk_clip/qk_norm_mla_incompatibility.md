# QK-Norm与MLA的不兼容性分析

## QK-Norm的基本原理

QK-Norm是Google在Gemma3中采用的技术，定义为：

$$\boldsymbol{O} = \text{softmax}(\tilde{\boldsymbol{Q}}\tilde{\boldsymbol{K}}{}^{\top})\boldsymbol{V}$$

其中：
$$\begin{aligned}
\tilde{\boldsymbol{Q}} &= \text{RMSNorm}(\boldsymbol{Q}) \\
\tilde{\boldsymbol{K}} &= \text{RMSNorm}(\boldsymbol{K})
\end{aligned}$$

**核心要求**：需要将$\boldsymbol{Q}$和$\boldsymbol{K}$完全**物化（Materialize）**出来，然后对它们分别进行RMSNorm标准化。

## MLA架构的特殊性

### MLA训练阶段 vs 推理阶段的差异

MLA（Multi-head Latent Attention）在训练和推理阶段使用**不同的计算方式**：

#### 训练/Prefill阶段
$$\begin{aligned}
\boldsymbol{q}_i^{(s)} &= \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\boldsymbol{\mathcal{R}}_i\right] \in \mathbb{R}^{d_k + d_r} \\
\boldsymbol{k}_i^{(s)} &= \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{kr}\boldsymbol{\mathcal{R}}_i\right] \in \mathbb{R}^{d_k + d_r} \\
\boldsymbol{v}_i^{(s)} &= \boldsymbol{c}_i\boldsymbol{W}_v^{(s)} \in \mathbb{R}^{d_v}
\end{aligned}$$

#### 推理/Decoding阶段
$$\begin{aligned}
\boldsymbol{q}_i^{(s)} &= \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)}\boldsymbol{W}_{kc}^{(s)\top}, \boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\boldsymbol{\mathcal{R}}_i\right] \in \mathbb{R}^{d_c + d_r} \\
\boldsymbol{k}_i &= \left[\boldsymbol{c}_i, \boldsymbol{x}_i\boldsymbol{W}_{kr}\boldsymbol{\mathcal{R}}_i\right] \in \mathbb{R}^{d_c + d_r} \\
\boldsymbol{v}_i &= \boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c \in \mathbb{R}^{d_c}
\end{aligned}$$

### 关键差异
注意推理阶段中：
- $\boldsymbol{k}_i$不再有上标$(s)$，即**所有头共享同一个K**
- $\boldsymbol{v}_i$不再有上标$(s)$，即**所有头共享同一个V**
- $\boldsymbol{q}_i^{(s)}$的维度从$d_k + d_r$变为$d_c + d_r$

## QK-Norm与MLA的冲突

### 1. 物化问题
**QK-Norm的要求**：
- 需要完整地物化出每个头的$\boldsymbol{Q}^{(s)}$和$\boldsymbol{K}^{(s)}$
- 然后对它们分别进行RMSNorm

**MLA的限制**：
- 在推理阶段，训练时的$\boldsymbol{K}^{(s)}$**根本不存在**
- 所有头共享同一个$\boldsymbol{K}$，无法分别处理

### 2. 存储和计算的不一致
**训练阶段**：
- 可以计算出每个头的$\boldsymbol{K}^{(s)} = \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{kr}\boldsymbol{\mathcal{R}}_i\right]$
- 理论上可以对其进行RMSNorm

**推理阶段**：
- KV Cache中存储的是$\boldsymbol{k}_i = \left[\boldsymbol{c}_i, \boldsymbol{x}_i\boldsymbol{W}_{kr}\boldsymbol{\mathcal{R}}_i\right]$
- **无法重构出训练时的$\boldsymbol{K}^{(s)}$**

### 3. 维度不匹配
- 训练时：$\boldsymbol{k}_i^{(s)} \in \mathbb{R}^{d_k + d_r}$
- 推理时：$\boldsymbol{k}_i \in \mathbb{R}^{d_c + d_r}$

由于$d_k \neq d_c$，即使想要强行进行某种形式的标准化，维度都对不上。

## 为什么MLA要这样设计？

### 1. 内存效率
- **训练时**：每个头都有独立的$\boldsymbol{K}^{(s)}$和$\boldsymbol{V}^{(s)}$
- **推理时**：所有头共享$\boldsymbol{K}$和$\boldsymbol{V}$，大幅减少KV Cache的内存占用

### 2. 计算效率
- 通过共享机制避免重复计算
- 在保持表达能力的同时提高推理效率

## 尝试的解决方案：Partial QK-Norm

博客中提到尝试过**Partial QK-Norm**：
- 对MLA中可以在推理时物化的部分（qr、qc、kr）进行RMSNorm
- kc部分在推理时无法完全重构，所以跳过

**结果**：
- ✅ 可以压制MaxLogit
- ❌ 长度外推效果非常糟糕

## 总结

QK-Norm不适用于MLA的根本原因是：

1. **架构不匹配**：QK-Norm需要完整物化Q、K矩阵，但MLA在推理时采用共享机制
2. **训练推理不一致**：MLA的训练和推理阶段使用不同的计算图
3. **维度不兼容**：训练和推理阶段的张量维度不同
4. **存储限制**：KV Cache无法存储训练时所需的完整信息

这正是为什么需要QK-Clip这样**直接作用于权重**而不改变前向计算的方法，来解决MLA架构下的MaxLogit爆炸问题。

---
*基于@qk_clip_blog.md中关于MLA与QK-Norm兼容性的分析整理*