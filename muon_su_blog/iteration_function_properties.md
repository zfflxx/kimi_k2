# 迭代函数的性质分析

## 迭代函数的定义

我们研究的核心函数是：
$$g(x) = ax + bx^3 + cx^5$$

或者用不动点参数化形式：
$$g(x) = x + \kappa x(x^2 - x_1^2)(x^2 - x_2^2)$$

这个函数描述了单个奇异值在Newton-Schulz迭代中的演化规律。

## 不动点分析

### 定义与求解
不动点是满足$g(x) = x$的点，即：
$$ax + bx^3 + cx^5 = x$$
$$(a-1)x + bx^3 + cx^5 = 0$$
$$x[(a-1) + bx^2 + cx^4] = 0$$

### 不动点的分类

#### 1. 显然不动点
- $x = 0$：总是不动点

#### 2. 非零不动点
需要求解：$(a-1) + bx^2 + cx^4 = 0$

用参数化形式更清晰：
$$g(x) = x + \kappa x(x^2 - x_1^2)(x^2 - x_2^2) = x$$
$$\kappa x(x^2 - x_1^2)(x^2 - x_2^2) = 0$$

显然有不动点：$x = 0, \pm x_1, \pm x_2$

### 不动点的稳定性

#### 线性化分析
不动点$x^*$的稳定性由$g'(x^*)$决定：
- $|g'(x^*)| < 1$：稳定（吸引）
- $|g'(x^*)| > 1$：不稳定（排斥）
- $|g'(x^*)| = 1$：边界情况

#### 导数计算
对于$g(x) = ax + bx^3 + cx^5$：
$$g'(x) = a + 3bx^2 + 5cx^4$$

在关键点的导数：
- $g'(0) = a$
- $g'(1) = a + 3b + 5c$
- $g'(-1) = a + 3b + 5c$

## 收敛域分析

### 定义
收敛域是所有收敛到目标不动点1的初始值$x_0$的集合：
$$\mathcal{C} = \{x_0 : \lim_{n \to \infty} g^{(n)}(x_0) = 1\}$$

其中$g^{(n)}$表示$g$的$n$次复合。

### 影响因素

#### 1. 不动点的相对位置
理想设计：$x_1 < 1 < x_2$，使得：
- 来自左侧的轨迹经过$x_1$向1收敛
- 来自右侧的轨迹经过$x_2$向1收敛

#### 2. 斜率控制
关键约束：
- $|g'(1)| < 1$：确保在目标点附近收敛
- $|g'(x_1)| < 1, |g'(x_2)| < 1$：确保中间不动点稳定

#### 3. 单调性区间
在$[x_1, x_2]$区间内，理想情况下$g$应该：
- 在$x_1$附近：$g(x) > x$（向右推动）
- 在$x = 1$：$g(1) = 1$（不动点）
- 在$x_2$附近：$g(x) < x$（向左推动）

## 收敛速度分析

### 线性收敛率
在不动点$x^* = 1$附近，收敛速度由$|g'(1)|$决定：
$$|x_{n+1} - 1| \approx |g'(1)| \cdot |x_n - 1|$$

### 最优收敛条件
为了最快收敛，我们希望：
$$g'(1) = a + 3b + 5c \to 0$$

但这与其他约束（如稳定性、收敛域大小）存在权衡。

### 超线性收敛的可能性
如果能设计$g'(1) = 0$且$g''(1) \neq 0$，则可能实现超线性收敛：
$$|x_{n+1} - 1| \approx \frac{|g''(1)|}{2} \cdot |x_n - 1|^2$$

## 数值稳定性考虑

### 1. 梯度控制
避免$|g'(x)|$在某些区域过大，防止：
- 数值溢出
- 敏感性放大
- 振荡行为

### 2. 有界性保证
确保$g$将有界区间映射到有界区间：
$$g: [-M, M] \to [-M', M']$$

其中$M, M'$是合理的边界。

### 3. 单调性区域
在关键收敛路径上保持适当的单调性，避免复杂的振荡模式。

## 实际奇异值分布的影响

### Marchenko-Pastur分布特性
对于随机矩阵，奇异值主要分布在$[0, 2]$区间内（归一化后）。

### 设计约束
需要确保$g$在这个区间内行为良好：
- $g(0) = 0$（保持零奇异值）
- 在$(0, 2]$内有效收敛到1
- 避免在此区间内出现不期望的不动点

## 多项式特性分析

### 5次多项式的灵活性
$g(x) = ax + bx^3 + cx^5$作为5次多项式：
- 最多有4个极值点
- 最多有4个实数不动点（除了0）
- 提供足够的自由度来精确控制收敛行为

### 奇函数性质
如果$a, b, c$选择使得$g(-x) = -g(x)$，则：
- 不动点关于原点对称
- 简化分析复杂度
- 但可能限制优化空间

## 参数敏感性分析

### 鲁棒性要求
优化得到的参数应该对小扰动不敏感：
$$\frac{\partial \text{收敛性能}}{\partial (a,b,c)} \text{ 应该适中}$$

### 硬件适应性
在不同数值精度（float32, bfloat16）下保持稳定：
- 避免接近机器精度的临界计算
- 容忍舍入误差的累积

## 可视化理解

### 迭代图分析
绘制$y = g(x)$和$y = x$的交点：
- 交点即为不动点
- $g$在$y = x$上方：$x$增大
- $g$在$y = x$下方：$x$减小

### 蛛网图（Cobweb Plot）
显示迭代轨迹：
1. 从$x_0$开始，垂直到$(x_0, g(x_0))$
2. 水平到$(g(x_0), g(x_0))$
3. 重复过程，观察收敛模式

### 收敛盆地
在复平面或扩展实轴上显示不同初值的收敛目标，揭示收敛域的几何结构。

## 实际应用指导

### 参数调优策略
1. **先固定目标**：确保$g(1) = 1$
2. **控制斜率**：调整$g'(1)$以平衡收敛速度和稳定性
3. **扩展收敛域**：调整$x_1, x_2$以覆盖预期的奇异值范围
4. **验证鲁棒性**：测试在不同初值分布下的表现

### 诊断工具
- **收敛测试**：在典型奇异值分布上测试收敛率
- **稳定性分析**：检查$|g'(x)|$在关键区域的值
- **边界行为**：验证在极端奇异值处的表现

这种系统性的分析为Newton-Schulz迭代的设计和优化提供了坚实的理论基础，也解释了为什么不同的系数选择会导致显著不同的性能表现。