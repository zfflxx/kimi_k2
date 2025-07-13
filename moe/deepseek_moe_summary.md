# DeepSeek MOE 原理概览总结

## 核心架构：DeepSeekMoE

### 基本设计思想
DeepSeekMoE相比传统MoE架构（如GShard）有两个关键创新：
1. **更细粒度的专家分割** - 提高专家专业化程度和知识获取精度
2. **共享专家隔离** - 减少路由专家间的知识冗余

### 数学公式
FFN输出计算：
$$\mathbf{h}_{t}^{\prime} = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} {\operatorname{FFN}^{(s)}_{i}\left( \mathbf{u}_{t} \right)} + \sum_{i=1}^{N_r} {g_{i,t} \operatorname{FFN}^{(r)}_{i}\left( \mathbf{u}_{t} \right)}$$

其中：
- $N_s$：共享专家数量，$N_r$：路由专家数量
- $g_{i,t}$：第$i$个专家的门控值
- $\operatorname{FFN}^{(s)}_{i}$：第$i$个共享专家，$\operatorname{FFN}^{(r)}_{i}$：第$i$个路由专家

## V2与V3的路由机制差异

### DeepSeek V2
- 使用**Softmax**计算亲和度分数：$s_{i,t} = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}}^{T} \mathbf{e}_{i} \right)$
- Top-K选择后直接使用亲和度分数作为门控值

### DeepSeek V3  
- 使用**Sigmoid**计算亲和度分数：$s_{i,t} = \operatorname{Sigmoid} \left( {\mathbf{u}_{t}}^{T} \mathbf{e}_{i} \right)$
- 对选中的亲和度分数进行**归一化**：$g_{i,t} = \frac{g^{\prime}_{i,t}}{\sum_{j=1}^{N_r} g^{\prime}_{j,t}}$

## 负载均衡策略

### V2的辅助损失方法
采用三种辅助损失：
1. **专家级平衡损失** $\mathcal{L}_{\mathrm{ExpBal}}$ - 防止路由崩塌
2. **设备级平衡损失** $\mathcal{L}_{\mathrm{DevBal}}$ - 确保设备间计算平衡
3. **通信平衡损失** $\mathcal{L}_{\mathrm{CommBal}}$ - 平衡设备间通信负载

### V3的无辅助损失策略
**创新性改进**：引入偏置项$b_i$动态调整路由：
$$g^{\prime}_{i,t} = \begin{cases} 
s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk} (\{ s_{j, t} + b_j | 1 \leq j \leq N_r \}, K_{r}), \\
0, & \text{otherwise}
\end{cases}$$

**动态调整机制**：
- 专家过载时：$b_i \leftarrow b_i - \gamma$
- 专家欠载时：$b_i \leftarrow b_i + \gamma$
- 偏置项仅用于路由选择，门控值仍基于原始亲和度分数

## 通信优化

### 设备限制路由（V2）
确保每个token最多分发到$M$个设备，通过以下步骤：
1. 选择包含最高亲和度专家的$M$个设备
2. 在这些设备上的专家中进行Top-K选择
3. 实践中$M \geq 3$时效果接近无限制路由

### 节点限制路由（V3）
类似机制，确保每个token最多发送到$M$个节点，实现计算-通信几乎完全重叠。

## Token处理策略

### V2的Token丢弃
- 采用设备级token丢弃策略
- 基于亲和度分数丢弃最低分token
- 保证约10%训练序列的token不被丢弃

### V3的无Token丢弃
由于有效的负载均衡策略，V3在训练和推理中都不丢弃token。

## 技术演进总结

DeepSeek从V2到V3的主要改进：
1. **路由机制**：Softmax → Sigmoid + 归一化
2. **负载均衡**：辅助损失 → 无辅助损失的偏置调整
3. **Token处理**：有丢弃 → 无丢弃
4. **性能提升**：更好的负载均衡与模型性能权衡

这些改进使得DeepSeek V3在保持计算效率的同时，实现了更好的模型性能和训练稳定性。