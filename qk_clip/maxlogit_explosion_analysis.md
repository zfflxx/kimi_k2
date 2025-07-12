# MaxLogit爆炸的原因分析

## 数学本质

MaxLogit爆炸的根本原因可以从以下不等式理解：

$$|\boldsymbol{q}_i\cdot \boldsymbol{k}_j| \leq \|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\| = \|\boldsymbol{x}_i\boldsymbol{W}_q\| \|\boldsymbol{x}_j\boldsymbol{W}_k\| \leq \|\boldsymbol{x}_i\| \|\boldsymbol{x}_j\| \|\boldsymbol{W}_q\| \|\boldsymbol{W}_k\|$$

由于输入$\boldsymbol{x}$通常经过RMSNorm标准化，$\|\boldsymbol{x}_i\| \|\boldsymbol{x}_j\|$相对稳定，因此：

**MaxLogit爆炸 ⟺ 权重矩阵$\boldsymbol{W}_q$和$\boldsymbol{W}_k$的谱范数趋向无穷大**

## 为什么Muon更容易引起爆炸？

### 1. 奇异值分布差异
- **Muon**：更新量经过$\text{msign}$运算，所有奇异值相等 → **满秩更新**
- **Adam**：更新量通常是低秩的，奇异值有大有小

### 2. "碰撞效应"
设参数$\boldsymbol{W}_{t-1}$的SVD为$\sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$：

**Muon更新**：
$$\boldsymbol{W}_t = \sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} + \sum_j \bar{\sigma}\bar{\boldsymbol{u}}_j \bar{\boldsymbol{v}}_j^{\top}$$

**Adam更新**：
$$\boldsymbol{W}_t = \sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} + \sum_j \tilde{\sigma}_j\tilde{\boldsymbol{u}}_j \tilde{\boldsymbol{v}}_j^{\top}$$

- Muon的满秩更新增加了与原参数奇异向量"碰撞"的概率
- 当奇异向量对齐时，奇异值直接叠加，导致权重矩阵谱范数增大

### 3. 双线性放大效应
注意力机制的双线性形式$\boldsymbol{q}_i\cdot \boldsymbol{k}_j = (\boldsymbol{x}_i \boldsymbol{W}_q)\cdot(\boldsymbol{x}_j \boldsymbol{W}_k)$：
- $\boldsymbol{W}_q$和$\boldsymbol{W}_k$的谱范数**连乘**，放大了爆炸风险
- 形成"糟的更糟"的恶性循环

## 为什么只有部分Head爆炸？

1. **奇异向量碰撞是小概率事件**：即使Muon增加了碰撞概率，完全对齐仍然罕见
2. **随机初始化差异**：不同Head的初始状态不同
3. **梯度传播差异**：不同Head接收到的梯度信号强度不同

## 大模型中的加剧因素

1. **参数量增大**：更多的参数维度增加了不稳定的可能性
2. **训练复杂度**：大模型训练中的各种不稳定因素相互耦合
3. **Weight Decay效果减弱**：在超大规模下难以有效控制所有参数

## 时间演化特征

- **初期**：权重随机初始化，MaxLogit相对稳定
- **中期**：优化器更新与权重增长形成"拉锯战"
- **后期**：在Weight Decay作用下，模型可能自发稳定

---
*基于@qk_clip_blog.md中的理论分析整理*