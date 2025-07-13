# Muon优化器概览总结

## 核心概念

Muon (MomentUm Orthogonalized by Newton-schulz) 是一个专为矩阵参数设计的优化器，声称比AdamW更高效。其核心思想是体现向量与矩阵的本质差异。

## 算法原理

### 更新规则
对于矩阵参数 $\boldsymbol{W}\in\mathbb{R}^{n\times m}$：

$$\begin{aligned}
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t [\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}]
\end{aligned}$$

### 矩阵符号函数 (msign)
不同于element-wise的sign函数，msign是矩阵化推广：
$$\text{msign}(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$$

其中 $\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}^{\top} = \text{SVD}(\boldsymbol{M})$

## 关键特性

### 自适应学习率特性
1. **损失函数缩放不变性**：损失乘以常数λ不影响优化轨迹
2. **各向同性更新**：通过SVD将不同奇异值都置为1，实现更均匀的更新幅度

### 不同参数类型的处理
- **对角矩阵**：退化为SignSGD/Tiger (element-wise sign)
- **向量** (视为$n\times 1$矩阵)：相当于$l_2$归一化
- **一般矩阵**：使用完整的矩阵符号函数

## 实现细节

### Newton-schulz迭代
为避免每步都做SVD，使用迭代近似：
$$\boldsymbol{X}_{t+1} = a\boldsymbol{X}_t + b\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2$$

作者给出的系数：$(a,b,c) = (3.4445, -4.7750, 2.0315)$

## 理论洞察

### 范数视角
Muon相当于**2-范数约束下的梯度下降**：
$$\boldsymbol{W}_{t+1} = \mathop{\text{argmin}}_{\boldsymbol{W}} \frac{\|\boldsymbol{W} - \boldsymbol{W}_t\|_2^2}{2\eta_t} + \mathcal{L}(\boldsymbol{W})$$

2-范数比F-范数更好地度量矩阵间的本质差异。

### 最优正交近似
当矩阵为方阵且满秩时：
$$\text{msign}(\boldsymbol{M}) = \mathop{\text{argmin}}_{\boldsymbol{O}^{\top}\boldsymbol{O} = \boldsymbol{I}}\|\boldsymbol{M} - \boldsymbol{O}\|_F^2$$

## 历史联系

### 与Shampoo的关系
Shampoo (2018) 缓存梯度的矩阵乘积 $\boldsymbol{G}\boldsymbol{G}^{\top}$ 和 $\boldsymbol{G}^{\top}\boldsymbol{G}$：
$$\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t \boldsymbol{L}_t^{-1/4}\boldsymbol{G}_t\boldsymbol{R}_t^{-1/4}$$

当$\beta=0$时，Shampoo与Muon理论等价。

### 考古发现
2015年的论文《Stochastic Spectral Descent for Restricted Boltzmann Machines》已提出类似算法。

## 实践考量

### 优势
- 计算开销仅增加约2-5%
- 显存使用比Adam更少（少一组缓存变量）
- 矩阵乘法可并行，不显著增加时间成本

### 挑战
- **非element-wise更新**：破坏了张量并行的简洁性
- **通信成本**：分布式训练时需要汇聚梯度
- **Multi-Head Attention问题**：需要将大矩阵拆分成小矩阵独立处理

## 核心思想

Muon的价值在于**认识到向量与矩阵的内在差异**：
- 传统优化器（SGD、Adam等）是element-wise的，将所有参数视为大向量
- Muon考虑矩阵的独有特性（如迹、特征值等），实现更本质的优化

这种差异化处理带来了更好的优化效果，但也引入了一些工程复杂性。