
四个月前，我们发布了Moonlight，在16B的MoE模型上验证了Muon优化器的有效性。在Moonlight中，我们确认了给Muon添加Weight Decay的必要性，同时提出了通过Update RMS对齐来迁移Adam超参的技巧，这使得Muon可以快速应用于LLM的训练。然而，当我们尝试将Muon进一步拓展到千亿参数以上的模型时，遇到了新的“拦路虎”——MaxLogit爆炸。

为了解决这个问题，我们提出了一种简单但极其有效的新方法，我们称之为“QK-Clip”。该方法从一个非常本质的角度去看待和解决MaxLogit现象，并且无损模型效果，这成为我们最新发布的万亿参数模型“Kimi K2”的关键训练技术之一。

## 问题描述 #

我们先来简单介绍一下MaxLogit爆炸现象。回顾Attention的定义
$$
\boldsymbol{O} = softmax(\boldsymbol{Q}\boldsymbol{K}^{\top})\boldsymbol{V}
$$
这里省略了缩放因子$1/\sqrt{d}$，因为它总可以吸收到$\boldsymbol{Q},\boldsymbol{K}$的定义中。“MaxLogit爆炸”中的Logit，指的是Softmax前的Attention矩阵，即$\boldsymbol{Q}\boldsymbol{K}^{\top}$，而MaxLogit指的是它取绝对值后的最大值，我们也可以写成
$$
S_{\max} = \Vert\boldsymbol{Q}\boldsymbol{K}^{\top}\Vert_{\infty} = \max_{i,j} |\boldsymbol{q}_i\cdot \boldsymbol{k}_j|
$$
这里的$\max$其实还要在batch_size维度上取，最终得到一个标量。而MaxLogit爆炸是指，$S_{\max}$随着训练的推进一直往上涨，增长速度是线性甚至是超线性的，并且在相当长的时间内没有稳定的迹象。

MaxLogit爆炸现象

MaxLogit本质上是一个异常值指标，它的爆炸意味着异常值超出了可控范围。具体来说，我们有
$$
|\boldsymbol{q}_i\cdot \boldsymbol{k}_j| \leq \Vert\boldsymbol{q}_i\Vert \Vert\boldsymbol{k}_j\Vert = \Vert\boldsymbol{x}_i\boldsymbol{W}_q\Vert \Vert\boldsymbol{x}_j\boldsymbol{W}_k\Vert \leq \Vert\boldsymbol{x}_i\Vert \Vert\boldsymbol{x}_j\Vert \Vert\boldsymbol{W}_q\Vert \Vert\boldsymbol{W}_k\Vert (1)
$$
由于$\boldsymbol{x}$通常会加RMSNorm，所以一般情况下$\Vert\boldsymbol{x}_i\Vert \Vert\boldsymbol{x}_j\Vert$是不会爆炸的，因此MaxLogit爆炸意味着谱范数$\Vert\boldsymbol{W}_q\Vert,\Vert\boldsymbol{W}_k\Vert$有往无穷大发展的风险，这显然不是一个好消息。

由于再大的数值经过Softmax后都变得小于1，所以比较幸运的情况下，这个现象不会带来太严重的后果，顶多是浪费了一个Attention Head，但比较糟糕的情况下，可能会引起Grad Spike甚至训练崩溃。因此，保险起见应当尽量避免MaxLogit爆炸的出现。

## 已有尝试 #

在《Muon续集：为什么我们选择尝试Muon？》中我们简单分析过，Weight Decay能一定程度上预防MaxLogit爆炸，所以小模型出现MaxLogit爆炸的概率很小，即便像Moonlight这样的16B模型，MaxLogit最多涨到120后就自动降下来了。

Moonlight的MaxLogit自动降了下来

换句话说，MaxLogit爆炸更多出现在非常大参数量的模型中，模型越大，训练的不稳定因素越多，Weight Decay越难稳定训练。这时候增加Weight Decay自然也能加强控制，但同时也会带来明显的效果损失，所以此路不通。另一个比较直接的思路是直接给Logit加$\text{softcap}$：
$$
\boldsymbol{O} = softmax(\text{softcap}(\boldsymbol{Q}\boldsymbol{K}^{\top};\tau))\boldsymbol{V}
$$
其中$\text{softcap}(x;\tau) = \tau\tanh(x/\tau)$，由Google的Gemma2引入。由于$\tanh$的有界性，$\text{softcap}$自然是能够保证$\text{softcap}$后的Logit有界的，但无法保证$\text{softcap}$前的Logit是有界的（亲测），所以$\text{softcap}$只是将一个问题转化为了另一个问题，实际上并没有解决问题。

也许Google自己都意识到了这一点，所以在后来的Gemma3中没有用$\text{softcap}$了，而改用“QK-Norm”：
$$
\boldsymbol{O} = softmax(\tilde{\boldsymbol{Q}}\tilde{\boldsymbol{K}}{}^{\top})\boldsymbol{V},\quad $$
\begin{aligned}

\tilde{\boldsymbol{Q}}=&\,\text{RMSNorm}(\boldsymbol{Q}) \\
\tilde{\boldsymbol{K}}=&\,\text{RMSNorm}(\boldsymbol{K})

\end{aligned}
$$
$$

QK-Norm确实是压制MaxLogit非常有效的方法，然而它只适用于MHA、GQA等，不适用于MLA，因为QK-Norm需要把$\boldsymbol{Q},\boldsymbol{K}$给Materialize出来，但对于MLA来说，它训练阶段跟Decoding阶段的$\boldsymbol{Q},\boldsymbol{K}$并不一样（如下式所示），在Decoding阶段我们没法完全Materialize训练阶段的$\boldsymbol{K}$，换言之，Decoding阶段没法做QK-Norm。

$$\require{cancel}$$
\begin{array}
{c|c}
\text{训练/Prefill} & \text{Decoding} \\
\\
\begin{gathered}
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt]
\boldsymbol{o}_t^{(s)} = \frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\[15pt]
\boldsymbol{q}_i^{(s)} = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)},\boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r}\\
\boldsymbol{k}_i^{(s)} = \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)},\boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r} \\
\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c}
\end{gathered}
&
\begin{gathered}
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}\boldsymbol{W}_v^{(1)}, \boldsymbol{o}_t^{(2)}\boldsymbol{W}_v^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\boldsymbol{W}_v^{(h)}\right] \\[10pt]
\boldsymbol{o}_t^{(s)} = \frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} }{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)} \\[15pt]
\boldsymbol{q}_i^{(s)} = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)}\boldsymbol{W}_{kc}^{(s)}{}^{\top}, \boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c + d_r}\\
\boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \left[\boldsymbol{c}_i, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c + d_r}\\
\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{c}_i= \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c}
\end{gathered} \\

\end{array}
$$

为什么要用MLA？我们已经用两篇文章《Transformer升级之路：21、MLA好在哪里?（上）》和《Transformer升级之路：21、MLA好在哪里?（下）》讨论了这个问题，这里不再重复。总之，我们希望MLA也能有类似QK-Norm的能够保证压制MaxLogit的手段。

## 直击目标 #

期间我们还尝试了一些间接手段，比如单独降低$\boldsymbol{Q},\boldsymbol{K}$的学习率、单独增大它们的Weight Decay等，但都不奏效。最接近成功的一次是Partial QK-Norm，对于MLA来说，它的$\boldsymbol{Q},\boldsymbol{K}$分为qr、qc、kr、kc四个部分，其中前三部分在Decoding时都是可以Materialize的，所以我们给这三部分都加上RMSNorm，结果是可以压制MaxLogit，但长度激活效果非常糟糕。

在失败多次之后，我们不禁开始反思：前面我们的尝试其实都只是压制MaxLogit的“间接手段”，真正能保证解决MaxLogit爆炸的直接手段是什么？从不等式$式(1)$我们不难联想到可以对$\boldsymbol{W}_q,\boldsymbol{W}_k$做奇异值裁剪，但这本质上还是间接手段，而且奇异值裁剪的计算成本也不低。

但很明显，对$\boldsymbol{W}_q,\boldsymbol{W}_k$进行事后缩放理论上是可行的，问题就是什么时候缩放、缩放多少。终于，某天福至心灵之下，笔者总算反应过来：MaxLogit本身就是触发缩放的最直接信号！具体来说，当MaxLogit超过期望阈值$\tau$时，我们直接给$\boldsymbol{Q}\boldsymbol{K}^{\top}$乘上$\gamma = \tau / S_{\max}$，那么新的MaxLogit肯定就不超过$\tau$了。乘$\gamma$的操作，我们可以分别吸收到权重$\boldsymbol{Q}\boldsymbol{K}$的权重上去，于是我们得到初版QK-Clip：
$$
\begin{aligned}

&\boldsymbol{W}_t = \text{Optimizer}(\boldsymbol{W}_{t-1}, \boldsymbol{G}_t) \\
&\text{if }S_{\max}^{(l)} > \tau\text{ and }\boldsymbol{W} \in \{\boldsymbol{W}_q^{(l)}, \boldsymbol{W}_k^{(l)}\}: \\
&\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \sqrt{\tau / S_{\max}^{(l)}}

\end{aligned}
$$

其中$S_{\max}^{(l)}$是第$l$层Attention的MaxLogit，$\boldsymbol{W}_q^{(l)}, \boldsymbol{W}_k^{(l)}$是它$\boldsymbol{Q},\boldsymbol{K}$的权重。也就是说，在优化器更新之后，根据$S_{\max}^{(l)}$的大小来决定是否对$\boldsymbol{Q},\boldsymbol{K}$的权重进行裁剪，裁剪的幅度直接由$S_{\max}^{(l)}$与阈值$\tau$的比例来决定，直接保证裁剪后的矩阵不再MaxLogit爆炸。同时，由于是直接对权重进行操作，所以不影响推理模式，自然也就兼容MLA了。

## 精细调整 #

初版QK-Clip确实已经能成功压制MLA的MaxLogit，但经过仔细观察模型的“内科”后，我们发现它会出现“过度裁剪”的问题，修复该问题后就得到最终版QK-Clip。

我们知道，不管哪种Attention变体都有多个Head，一开始我们是每一层Attention只监控一个MaxLogit指标，所有Head的Logit是放在一起取Max的，这导致QK-Clip也是所有Head一起Clip的。然而，当我们分别监控每个Head的MaxLogit后发现，实际上每层只有为数不多的Head会出现MaxLogit爆炸，如果所有Head按同一个比例来Clip，那么大部份Head都是被“无辜受累”的了，这就是过度裁剪的含义。

所以，为了避免“殃及池鱼”，我们应该Per-Head地进行监控MaxLogit和QK-Clip。不过这里边又隐藏了另一个魔鬼细节：初版QK-Clip是将Clip因子平摊到$\boldsymbol{Q},\boldsymbol{K}$上的，但是MLA的$\boldsymbol{Q},\boldsymbol{K}$有qr、qc、kr、kc四部分，其中kr是所有Head共享的，如果对它Clip，那么同样会有“殃及池鱼”的问题。因此，对于(qr, kr)，我们应该只Clip到qr上去。

经过上述调整，最终版的QK-Clip为
$$
\begin{aligned}

&\boldsymbol{W}_t = \text{Optimizer}(\boldsymbol{W}_{t-1}, \boldsymbol{G}_t) \\
&\text{if }S_{\max}^{(l,h)} > \tau: \\
&\qquad\text{if }\boldsymbol{W} \in \{\boldsymbol{W}_{qc}^{(l,h)}, \boldsymbol{W}_{kc}^{(l,h)}\}: \\
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \sqrt{\tau / S_{\max}^{(l,h)}} \\
&\qquad\text{elif }\boldsymbol{W} \in \{\boldsymbol{W}_{qr}^{(l,h)}\}: \\
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \tau / S_{\max}^{(l,h)}

\end{aligned}
$$
其中上标${}^{(l,h)}$表示第$l$层、第$h$个Head。

## 扩展之路 #

至此，QK-Clip的操作细节已经介绍完毕，它直接以我们期望的MaxLogit为信号，对$\boldsymbol{Q},\boldsymbol{K}$的权重进行尽可能小的改动，达到了将MaxLogit值控制在指定阈值内的效果。同时因为这是直接对权重进行修改的方法，所以它兼容性比QK-Norm更好，可以用于MLA。

在Kimi K2的训练中，我们设置阈值$\tau$为100，总训练步数约为220k steps，从大致7k steps开始，就出现了MaxLogit超过$\tau$的Head，此后在相当长的时间内，Muon Update和QK-Clip都在“拉锯战”，即Muon想要增加MaxLogit而QK-Clip想要降低MaxLogit，它们一直处于微妙的平衡状态。有趣的是，70k steps之后，所有Head的MaxLogit都主动降低到了100以下，QK-Clip不再生效。

经过接近70k steps的Muon和QK-Clip拉锯战后，MaxLogit 主动降了下来

这表明，在Weight Decay的作用下，只要我们能稳住训练，模型最后很可能都会主动将MaxLogit降下来，QK-Clip的作用，正是帮助模型更平稳地度过训练初期。可能有读者担心QK-Clip会有损效果，但我们在小模型上做了对比实验，即便通过QK-Clip将MaxLogit压得特别小（比如30），也没有观察到效果有实质区别，再加上中后期模型会主动将MaxLogit降下来的这一现象，我们有理由相信QK-Clip对效果是无损的。

我们在实验中也观察到，Muon普遍比Adam更容易MaxLogit爆炸，所以某种程度上来说，QK-Clip是专门为Muon补充的更新规则，它是Muon在超大规模训练上的“通关秘笈”之一，这也是本文标题的含义。为此，我们将我们Moonlight中所提的Muon改动跟QK-Clip组合起来，起了个“MuonClip”的名字（$\boldsymbol{W}\in\mathbb{R}^{n\times m}$）：
$$\text{MuonClip}\quad\left\{\quad$$
\begin{aligned}

&\boldsymbol{M}_t = \mu \boldsymbol{M}_{t−1} + \boldsymbol{G}_t \\[8pt]
&\boldsymbol{O}_t = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{M}_t) \underbrace{\times \sqrt{\max(n,m)}\times 0.2}_{\text{Match Adam Update RMS}} \\[8pt]
&\boldsymbol{W}_t = \boldsymbol{W}_{t−1} − \eta_t (\boldsymbol{O}_t + \lambda \boldsymbol{W}_{t-1}) \\[8pt]
&\left.\begin{aligned}
&\text{if }S_{\max}^{(l,h)} > \tau: \\
&\qquad\text{if }\boldsymbol{W} \in \{\boldsymbol{W}_{qc}^{(l,h)}, \boldsymbol{W}_{kc}^{(l,h)}\}: \\
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \sqrt{\tau / S_{\max}^{(l,h)}} \\
&\qquad\text{elif }\boldsymbol{W} \in \{\boldsymbol{W}_{qr}^{(l,h)}\}: \\
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \tau / S_{\max}^{(l,h)}

\end{aligned}
$$\quad\right\} \text{QK-Clip}
\end{aligned}\right.$$

注意，“Muon普遍比Adam更容易MaxLogit爆炸”并不意味着只有Muon会MaxLogit爆炸，我们知道DeepSeek-V3是Adam训练的，而我们从DeepSeek-V3的开源模型中也观察到了MaxLogit爆炸现象，还有Gemma2用$\text{softcap}$防止MaxLogit爆炸，它也是Adam训的。因此，虽然我们强调了QK-Clip对Muon的价值，但如果读者坚持用Adam，那么它也是可以跟Adam结合成AdamClip的。

## 原因思考 #

为什么Muon更容易导致MaxLogit爆炸呢？这一节笔者尝试给出一个理论角度的解释，供大家参考。

从不等式$式(1)$可以看出，MaxLogit爆炸往往意味着$\boldsymbol{W}_q$或$\boldsymbol{W}_k$的谱范数出现爆炸的迹象，实际上谱范数的定义中也包含了取$\max$操作，两者本质上是相通的。因此，问题可以转化为“为什么Muon更容易导致谱范数爆炸”。我们知道谱范数等于最大的奇异值，所以又可以进一步联想到“为什么Muon更倾向于增大奇异值”。

Muon和Adam的区别是什么呢？Muon给出的更新量是经过$\msign$运算的，所有奇异值都相等，即它的有效秩是满秩；而一般情况下的矩阵，奇异值通常都是有大有小，并且以前面几个奇异值为主，从有效秩的角度看它们是低秩的，我们对Adam更新量的假设也是如此。这个假设并不新鲜，比如高阶muP同样假设了Adam更新量的低秩性。

用公式来说，我们设参数$\boldsymbol{W}_{t-1}$的SVD为$\sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$，Muon更新量的SVD为$\sum_j \bar{\sigma}\bar{\boldsymbol{u}}_j \bar{\boldsymbol{v}}_j^{\top}$，Adam更新量的SVD为$\sum_j \tilde{\sigma}_j\tilde{\boldsymbol{u}}_j \tilde{\boldsymbol{v}}_j^{\top}$，那么
\begin{gather}
\boldsymbol{W}_t = \sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} + \sum_j \bar{\sigma}\bar{\boldsymbol{u}}_j \bar{\boldsymbol{v}}_j^{\top}\qquad (\text{Muon}) \\
\boldsymbol{W}_t = \sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} + \sum_j \tilde{\sigma}_j\tilde{\boldsymbol{u}}_j \tilde{\boldsymbol{v}}_j^{\top}\qquad (\text{Adam}) \\
\end{gather}

很明显，如果奇异向量对$\boldsymbol{u}_i \boldsymbol{v}_i^{\top}$跟某个$\bar{\boldsymbol{u}}_j \bar{\boldsymbol{v}}_j^{\top}$或$\tilde{\boldsymbol{u}}_j \tilde{\boldsymbol{v}}_j^{\top}$很接近，那它们将会直接叠加起来，从而增大$\boldsymbol{W}_t$的奇异值。由于Muon的更新量是满秩的，所以它与$\boldsymbol{W}_{t-1}$的“碰撞几率”会远大于Adam的，所以Muon更容易增大参数的奇异值。

当然，上述分析是通用的，不限于$\boldsymbol{Q},\boldsymbol{K}$的权重，实际上在Moonlight中我们已经验证过，Muon训出来的模型权重的奇异值熵普遍更高，这也佐证了上述猜测。Attention Logit的特殊之处在于，它是双线性形式$\boldsymbol{q}_i\cdot \boldsymbol{k}_j = (\boldsymbol{x}_i \boldsymbol{W}_q)\cdot(\boldsymbol{x}_j \boldsymbol{W}_k)$，$\boldsymbol{W}_q,\boldsymbol{W}_k$的连乘使得爆炸的风险更大，还容易导致“糟的更糟”的恶性循环，最终促成了MaxLogit爆炸。

Muon与Adam训练的模型权重奇异值熵（等价于有效秩）比较

最后就是“Muon的碰撞几率远大于Adam”是相对而言的，实际上奇异向量碰撞在一块还是小概率事件，这也就能解释为啥只有小部份Attention Head会有MaxLogit爆炸现象了。

## 一些延伸 #

写到这里，关于QK-Clip比较重要计算和实验细节应该都讲清楚了。另外还需要提醒的是，QK-Clip思想很简单，但由于需要Per-Head来Clip，因此在分布式训练中写起来还是略微有点难度的，因为此时的参数矩阵往往被切分得“支离破碎”（在Muon基础上改起来不算难，在Adam基础上改则稍显复杂）。

对于笔者及其团队来说，QK-Clip不单单是解决MaxLogit爆炸问题的一个具体方法，还是反复尝试通过间接手段来解决问题且失败后的一次”幡然醒悟“：既然有了明确的度量指标，那么我们应该寻求能够保证解决问题的直接思路，而不是在降低LR、增大Weight Decay、部分QK-Norm等可能但不一定能解决问题的思路上浪费时间。

从方法上来看，QK-Clip的思路也不限于解决MaxLogit爆炸，它可以说是解决很多训练不稳定问题的“抗生素”。所谓抗生素，指的是它也许并不是解决问题最精妙的方法，但往往是解决问题最直接有效的方法之一，QK-Clip正是具有这个特点，它可以一般地推广成“哪里不稳Clip哪里”。

比如，有些情况下模型会出现“MaxOutput爆炸”的问题，这时候我们可以考虑根据MaxOutput的值来Clip权重$\boldsymbol{W}_o$。类比QK-Clip的Per-Head操作，这里我们也需要考虑Per-Dim操作，但Per-Dim Clip的成本显然太大，可能需要折中一下。总之，“哪里不稳Clip哪里”提供了统一的解决思路，但具体细节就要看大家发挥了。

## 文章小结 #

本文提出了QK-Clip，它是MaxLogit爆炸问题的一种新思路，跟QK-Norm不同，它是对Q、K权重的一种事后调整方案，并不改变模型的前向计算，因此适用性更广，它是“Muon + MLA”组合在超大规模训练上的重要稳定策略，也是我们最新发布的万亿模型Kimi K2的关键技术之一。

转载自：https://kexue.fm/archives/11126