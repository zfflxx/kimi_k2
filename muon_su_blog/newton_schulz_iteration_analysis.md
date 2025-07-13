# Newton-Schulz迭代求解深度解析

## 问题背景

在Muon优化器中，我们需要计算矩阵符号函数$\text{msign}(\boldsymbol{M})$。虽然可以通过SVD精确计算：

$$\text{msign}(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$$

但SVD计算成本高昂，特别是在深度学习的大规模矩阵场景下。因此需要更高效的近似方法。

## 理论出发点

### 基础恒等式
Newton-Schulz迭代基于矩阵符号函数的恒等式：

$$\text{msign}(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}$$

这个等式将问题转化为计算$(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}$。

### 泰勒展开策略
不失一般性假设$n \geq m$，在$\boldsymbol{M}^{\top}\boldsymbol{M} = \boldsymbol{I}$处对$(t)^{-1/2}$进行泰勒展开：

$$t^{-1/2} = 1 - \frac{1}{2}(t-1) + \frac{3}{8}(t-1)^2 - \frac{5}{16}(t-1)^3 + \cdots$$

保留到二阶项，得到：
$$t^{-1/2} \approx \frac{15 - 10t + 3t^2}{8}$$

## 理论迭代公式

将标量函数的泰勒展开直接应用到矩阵：

$$\text{msign}(\boldsymbol{M}) \approx \frac{15}{8}\boldsymbol{M} - \frac{5}{4}\boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M}) + \frac{3}{8}\boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^2$$

由此得到迭代格式（设$\boldsymbol{X}_t$是$\text{msign}(\boldsymbol{M})$的近似）：

$$\boldsymbol{X}_{t+1} = \frac{15}{8}\boldsymbol{X}_t - \frac{5}{4}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + \frac{3}{8}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2$$

**理论系数**：$(a, b, c) = (15/8, -5/4, 3/8) = (1.875, -1.25, 0.375)$

## 官方实现的差异

然而，Muon官方代码使用的系数是：
$$(a, b, c) = (3.4445, -4.7750, 2.0315)$$

这与理论推导的系数明显不同！

## 优化后的系数推导

### 一般化迭代形式
考虑更一般的三项迭代：

$$\boldsymbol{X}_{t+1} = a\boldsymbol{X}_t + b\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2$$

### 奇异值迭代分析
通过SVD分析可以证明，上述迭代等价于对每个奇异值$\sigma$进行迭代：

$$\sigma_{t+1} = g(\sigma_t) = a\sigma_t + b\sigma_t^3 + c\sigma_t^5$$

目标是让所有奇异值收敛到1。

### 重新参数化
将$g(x)$重新参数化为：

$$g(x) = x + \kappa x(x^2 - x_1^2)(x^2 - x_2^2)$$

其中$x_1 \leq x_2$是两个设计参数，对应迭代的不动点。

这样参数化的优势：
- 直观表示不动点：$0, \pm x_1, \pm x_2$
- 选择$x_1 < 1 < x_2$，使得迭代向1收敛

### 优化目标
将系数选择视为优化问题：

**目标**：最小化收敛误差
$$\min_{\kappa, x_1, x_2} \mathbb{E}[(g^{(T)}(\sigma_0) - 1)^2]$$

其中$g^{(T)}$表示迭代$T$次，$\sigma_0$是初始奇异值分布。

## 实验结果分析

博客中给出的优化结果表明：

### 关键发现：
1. **矩阵形状依赖**：最优系数与矩阵的$n, m$相关
2. **迭代步数敏感**：$T=3$和$T=5$的最优系数差异很大
3. **非方阵优势**：非方阵比方阵更容易收敛
4. **官方系数来源**：Muon作者的系数大致对应$T=5$时方阵的最优解

### 典型结果（$n=m=1024, T=5$）：
- 优化后：$(a, b, c) = (3.297, -4.136, 1.724)$，MSE = 0.02733
- 官方系数的MSE = 0.04431

## 算法实现要点

### 初始化策略
$$\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\|\boldsymbol{M}\|_F}$$

选择F-范数归一化的原因：
- 不改变SVD的$\boldsymbol{U}, \boldsymbol{V}$
- 将所有奇异值缩放到$[0,1]$区间
- 提供更稳定的迭代起点

### 计算复杂度
每次迭代需要：
- 2次矩阵乘法：$\boldsymbol{X}_t^{\top}\boldsymbol{X}_t$
- 3次矩阵乘法：构造更新项

总共5次矩阵乘法，对于$T=5$次迭代，需要25次矩阵乘法。

### 并行化优势
- 矩阵乘法是静态大小，易于并行
- 发生在梯度计算间隙，充分利用闲置算力
- 显存开销比Adam更小（少一组缓存变量）

## 深层理解

### 为什么Newton-Schulz有效？
1. **收敛性**：在适当条件下保证收敛到正交矩阵
2. **稳定性**：比直接SVD数值更稳定
3. **可调性**：通过优化系数适应不同场景

### 与其他方法的比较
- **vs SVD**：计算更快，但精度略低
- **vs 幂迭代**：收敛更快，更稳定
- **vs 其他迭代**：Newton-Schulz有理论保证和良好的数值特性

## 实际应用建议

### 系数选择策略：
1. **通用场景**：使用官方系数$(3.4445, -4.7750, 2.0315)$
2. **特定应用**：根据主要矩阵尺寸优化系数
3. **计算受限**：减少迭代步数$T$，相应调整系数

### 收敛监控：
可以通过检查$\|\boldsymbol{X}_t^{\top}\boldsymbol{X}_t - \boldsymbol{I}\|_F$来监控收敛情况。

这种迭代方法巧妙地平衡了计算效率和精度，是Muon优化器能够实用化的关键技术。