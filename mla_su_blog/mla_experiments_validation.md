# 实验如何验证这些猜测？

## 实验设计的核心原则

### 控制变量法
每个实验都遵循**单一变量控制**原则，确保能够分离各个因素的独立作用。

### 公共实验设置
所有实验统一使用：
- 类似LLAMA3的Dense模型
- hidden_size=2048, num_layers=12, num_heads=16
- 训练长度4096, 总tokens=16B, 训练步数=16k
- 优化器：Muon，Attention部分per head更新

## 验证猜测1：增大head_dims的作用

### Part I实验设计
**对比组合**：
- MLA（576 Cache）vs GQA2-128（512 Cache）vs GQA1-256（512 Cache）
- 增加GQA1-256-PR验证Partial RoPE

**验证逻辑**：
```
如果head_dims是关键 → GQA1-256应该优于GQA2-128
实际结果：GQA1-256 > GQA2-128 ✓
```

### Part II进一步验证
**扩展对比**：
- 增加MHA（head_dims=128）
- 增加GQA2-192（head_dims=192）  
- 增加MLA-256（head_dims=192+64）

**关键发现**：
```
Loss排序：GQA2-128 > GQA2-192 > GQA1-256
结论：增加head_dims比增加num_groups更有效 ✓
```

### Part VI参数量对齐验证
**实验变种**：
1. **double-heads**：GQA2-128的num_heads翻倍
2. **缩减MLP**：缩小intermediate_size  
3. **Q&O LoRA**：对Query和Output矩阵使用LoRA

**验证结果**：
```
heads翻倍 vs head_dims翻倍：Loss差0.003（head_dims更优）
缩小MLP vs 减小head_dims：Loss优0.004（head_dims重要性确认）
Q&O LoRA：在参数量几乎不增的情况下，head_dims翻倍仍有显著收益
```

## 验证猜测2：KV-Shared的作用

### Part III & IV实验设计

**KV-Shared方案设计**：

#### S1方案（部分共享）
- 192维：NoPE，K、V间共享
- 64维：RoPE，只用于K
- V额外投影64维，拼接到共享的192维
- 总Cache：(192+64+64)*2=640

#### S2方案（完全共享）  
- 192维：NoPE，K、V完全共享
- 64维：RoPE，K、V完全共享
- 引入VO-RoPE保证相对位置编码
- 总Cache：(192+64)*2=512

**验证结果**：
```
GQA2-(192+64)-S1：Loss=2.714（超过基准MLA）
GQA2-(192+64)-S2：Loss=2.708（进一步提升）
结论：KV-Shared确实有效，完全共享优于部分共享 ✓
```

### Part V扩展验证
**多种配置对比**：
- GQA4-(64+64)-S2, GQA4-(128+64)-S2
- GQA1-(512+64)-S3（类似MLA推理形式）

**关键发现**：
```
1. KV-Shared的GQA自带Partial RoPE效果
2. 同等Cache下，head_dims越大越好
3. KV-Shared提供了head_dims提升的天花板
```

## 验证猜测3：Partial RoPE的作用

### 直接对比验证
**实验组**：
- GQA1-256（标准RoPE）
- GQA1-256-PR（Partial RoPE：192维NoPE + 64维RoPE）

**结果**：
```
GQA1-256-PR > GQA1-256
Loss差异：2.711 vs 2.72
结论：Partial RoPE确实有正面作用 ✓
```

### 间接验证
**观察**：所有KV-Shared的方案都自然包含了Partial RoPE设计
**结果**：这些方案普遍表现良好，间接支持了Partial RoPE的有效性

## 实验验证的层次结构

### 第一层：单因素验证
每个猜测都有独立的对比实验：
- head_dims: GQA1-256 vs GQA2-128
- KV-Shared: S1/S2 vs 标准GQA
- Partial RoPE: PR vs 非PR版本

### 第二层：交叉验证  
不同实验之间的结果相互印证：
- MLA-256 > MLA：支持head_dims重要性
- S2 > S1：支持完全KV共享
- 所有KV-Shared方案的成功：支持Partial RoPE

### 第三层：参数对齐验证
排除参数量差异的干扰：
- Q&O LoRA实验证明head_dims的本质重要性
- 不是简单的参数增加，而是架构优势

## 实验设计的科学性

### 控制变量严格
- 相同的训练设置和数据
- 只改变Attention架构
- 系统性的ablation study

### 对比基准合理
- 选择了具有代表性的baseline
- Cache大小基本对齐
- 覆盖了主要的设计空间

### 结果分析深入
- 不仅看最终Loss，还分析训练过程
- 考虑了实际工程因素（如TP能力）
- 提供了理论解释

## 验证结论的可靠性

通过这套系统性实验，作者成功验证了：
1. **head_dims增大**：效果最显著，是最重要的因素
2. **Partial RoPE**：有正面作用，且与其他优化兼容
3. **KV-Shared**：有一定作用，特别是在大head_dims场景下

这些验证为理解MLA的成功提供了坚实的实证基础。