# 实验章节详细解释

## 实验设置 (Experimental Setups)

### 1. 模型架构 (Model Architecture)

**选择DeepSeekMoE架构的原因：**
- DeepSeekMoE相比传统的GShard架构有显著性能提升
- 它将专家分割为更细粒度的单元，提供更高的专家专业化
- 引入了一些共享专家来减少路由专家间的知识冗余

**架构细节：**
- 1B参数模型：9个MoE层，64个路由专家，激活6个专家
- 3B参数模型：11个MoE层，64个路由专家，激活6个专家
- 都有2个共享专家

**门控函数选择：**
- 主实验选择sigmoid而非softmax作为门控函数 $G$
- 实验发现sigmoid基线比softmax基线性能更好
- softmax的结果放在附录中讨论

### 2. 训练设置 (Training Settings)

**数据集：**
- 使用DeepSeek-AI创建的多语言训练语料库
- 包含网络文本、数学材料、编程脚本和出版文献等多样化文本
- 使用BPE分词器，词汇量32K

**训练规模：**
- 1B模型：训练100B tokens
- 3B模型：训练200B tokens
- 这个规模确保了充分的训练

**学习率调度：**
- 1B模型：余弦学习率调度器（1e-3 → 1e-4）
- 3B模型：多阶段学习率调度器

### 3. 基线比较 (Baseline)

**选择的基线：**
- 传统的辅助损失控制方法
- 辅助损失系数 $\alpha = 0.001$
- 这个值通过Figure 2的分析确定，在负载均衡和模型性能间取得合理平衡

**为什么不比较Expert Choice：**
- EC方法存在未来token泄露问题
- 这个问题在4.3节中详细讨论

### 4. 评估指标 (Metrics)

**性能指标：**
- 困惑度 (Perplexity)：在验证集上评估

**负载均衡指标 - MaxVio：**
$$\text{MaxVio} = \frac{\max_i \text{Load}_{i} - \overline{\text{Load}_{i}}}{\overline{\text{Load}_{i}}}$$

**两种MaxVio变体：**
1. **$\text{MaxVio}_\text{global}$：**
   - 在整个验证集上统计 $\text{Load}_{i}$
   - 反映专家利用的均衡程度
   - 体现批次大小接近极限时的效率上界

2. **$\text{MaxVio}_\text{batch}$：**
   - 在每个训练批次上统计 $\text{Load}_{i}$
   - 更直接关联训练效率
   - 反映实际训练过程中的负载均衡

## 主要结果 (Main Results)

### 核心发现

**Table 1的结果分析：**

| 模型规模 | 方法 | 验证困惑度 | $\text{MaxVio}_\text{global}$ |
|---------|------|------------|------------------------------|
| 1B | Loss-Controlled | 9.56 | 0.72 |
| 1B | Loss-Free | **9.50** | **0.04** |
| 3B | Loss-Controlled | 7.97 | 0.52 |
| 3B | Loss-Free | **7.92** | **0.04** |

**关键观察：**
1. **性能提升：** Loss-Free方法在两个模型规模上都获得更低的困惑度
2. **负载均衡显著改善：** $\text{MaxVio}_\text{global}$从0.72降至0.04（1B），从0.52降至0.04（3B）
3. **一致性：** 无论模型规模如何，改进都很一致

### 训练过程分析

**Figure 3的训练曲线：**
- 展示了整个训练过程中的 $\text{MaxVio}_\text{batch}$ 变化
- Loss-Free方法在大部分训练时间内保持更好的负载均衡
- 为了可视化，对100个相邻步骤进行了平均

**突破性意义：**
- 打破了负载均衡与模型性能间的困境
- 证明了无辅助损失的方法可以同时实现更好的性能和负载均衡

## 偏置更新算法的实证研究 (Empirical Studies)

### 1. 更新率研究 (Update Rate)

**更新率 $u$ 的作用：**
- 控制专家偏置 $\{b_i\}_{i=1}^N$ 收敛到"合适偏置"的速度

**不同更新率的效果：**
- **$u = 0.0001$（过低）：** 收敛缓慢，早期训练负载均衡差
- **$u = 0.01$（过高）：** 训练后期偏置波动，负载均衡恶化
- **$u = 0.001$（最优）：** 良好的训练均衡和验证困惑度

**Figure 4的分析：**
- 展示了不同更新率对训练负载均衡的影响
- 低更新率在训练早期表现差
- 高更新率在训练后期表现差
- 验证了 $u = 0.001$ 是最优选择

### 2. 更新规则研究 (Update Rule)

**两种更新规则对比：**
1. **原始版本：** $b_i = b_i + u \cdot \text{sign}(e_i)$
2. **变体版本：** $b_i = b_i + u \cdot e_i$

**实验结果（Table 2）：**
- 变体版本在负载均衡上略有改善
- 但在模型性能上没有提升
- 因此保持使用sign版本

**原因分析：**
- 变体版本会导致更大的波动
- 无法像sign版本那样维持正/负误差专家集合内的选择顺序

### 3. 乘法偏置研究 (Multiplicative Bias)

**乘法偏置的定义：**
$$g_{i, t} = \begin{cases}s_{i, t}, & s_{i, t} \times b_i \in \text{Topk}(\{s_{j, t} \times b_j\}) \\ 0, & \text{otherwise} \end{cases}$$

**与加法偏置的对比：**
- 初始化为1而非0
- 更新过程类似Algorithm 1

**实验结果（Table 3）：**
- 乘法偏置在负载均衡上表现相似
- 但在模型性能上略差
- 因此选择加法偏置

**原因分析：**
- 乘法偏置更容易影响高分专家的相对顺序
- 它改变的是比例而非绝对分数差距

## 实验设计的深层思考

### 1. 实验的系统性
- 从1B到3B的扩展验证了方法的可扩展性
- 充分的训练token数确保了结论的可靠性
- 多个超参数的消融研究保证了最优配置

### 2. 基线选择的合理性
- 选择0.001的辅助损失系数是基于Figure 2的权衡分析
- 避免了过强或过弱的辅助损失带来的偏见

### 3. 指标设计的科学性
- MaxVio指标直观反映负载不均衡程度
- 全局和批次级别的区分有助于理解不同层面的均衡

### 4. 实验的局限性
- 只在DeepSeekMoE架构上验证
- 数据集主要是多语言文本，缺乏其他领域验证
- 模型规模相对较小（最大3B）

这些实验结果强有力地证明了Loss-Free方法的有效性，不仅在理论上合理，在实践中也表现出色。