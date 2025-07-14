### 来读一下Muon优化器

- @muon.note.md @muon_su_blog.md

- 帮我概览的总结一下介绍Muon优化器的这篇博客
  - [Muon优化器概览总结](muon_su_blog/muon_optimizer_summary.md)
  - 为什说核心思想是体现向量与矩阵的本质差异?
    - [向量与矩阵的本质差异解析](muon_su_blog/vector_matrix_difference.md)
  - 给我详细深入的解释矩阵符号函数
    - [矩阵符号函数深入解析](muon_su_blog/matrix_sign_function_detailed.md)
  - $\boldsymbol{M}_t$ 是什么?
    - [动量矩阵详解](muon_su_blog/momentum_matrix_explanation.md)
  - 为什么要使用与矩阵符号函数的协同, 而不是直接使用动量矩阵呢?
    - [为什么需要矩阵符号函数](muon_su_blog/why_msign_with_momentum.md)
  - 帮我深入理解 F-范数 vs 2-范数
    - [F-范数vs2-范数深入对比](muon_su_blog/frobenius_vs_spectral_norm.md)

- 为什么优化器要有这两个特性: 1、损失函数的常数缩放不影响优化轨迹；2、每个参数分量的更新幅度尽可能一致?
  - [优化器的两个核心特性分析](muon_su_blog/optimizer_characteristics_analysis.md)
  - 对于每个参数分量的更新幅度尽可能一致, 难道不应该就根据梯度大小来更新吗, 如果幅度都一致了, 是不是重要的方向和不重要的方向都在无差异的更新?
    - [均匀更新幅度的悖论分析](muon_su_blog/uniform_update_paradox.md)
  - 给我详细解释一下标准化尺度在adamw中是如何实现的?
    - [AdamW中的标准化尺度机制详解](muon_su_blog/adamw_scale_normalization.md)

- 帮我理解符号函数小节
  - [符号函数小节深度解析](muon_su_blog/sign_function_analysis.md)
  - 如何理解msign(M)还有一个意义是"最优正交近似"?
    - [msign作为最优正交近似的深度理解](muon_su_blog/msign_orthogonal_approximation.md)
  - 帮我理解: 我们都可以视为对更新量的一种规整化约束，所以Muon和Signum、Tiger可以视作是同一思路下的优化器，它们都以动量M为出发点来构建更新量，只是为更新量选择了不同的规整化方法
    - [优化器的统一规整化框架](muon_su_blog/optimizer_regularization_framework.md)

- 帮我理解迭代求解小节
  - [Newton-Schulz迭代求解深度解析](muon_su_blog/newton_schulz_iteration_analysis.md)
  - 为什么通过泰勒展开算的理论系数和Muon官方代码使用的系数不一样, 官方的代码在[muon.py](muon_su_blog/muon.py)中
    - [理论系数与官方系数差异的深度分析](muon_su_blog/theory_vs_practice_coefficients.md)

- 帮我深入理解收敛加速这小节
  - [收敛加速小节深度解析](muon_su_blog/convergence_acceleration_analysis.md)
  - 帮我理解迭代函数的性质分析
    - [迭代函数的性质分析](muon_su_blog/iteration_function_properties.md)

- 帮我深入理解一些思考这小节
  - [Muon设计思考的深度解析](muon_su_blog/muon_reflections_analysis.md)

- 帮我深入理解范数视角这小节
  - [范数视角深度解析](muon_su_blog/norm_perspective_analysis.md)
  - 那adamw这种优化器放在这个范数视角是什么样子的
    - [AdamW的范数视角深度分析](muon_su_blog/adamw_norm_perspective.md)

- 帮我深入理解矩阵范数这小节
  - [矩阵范数深度解析](muon_su_blog/matrix_norm_analysis.md)
  - 为什么矩阵的梯度可以用Frobenius内积表示?

### 读Muon is Scalable for LLM Training

- @muon_scalable.note.md @muon_scalable.tex

- 帮我概览的总结一下这篇论文
  - [论文总结](/muon_su_blog/muon_scalable_summary.md)
