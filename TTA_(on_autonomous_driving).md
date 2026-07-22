# TENT: FULLY TEST-TIME ADAPTATION BY ENTROPY MINIMIZATION

## 1. Problem

- 模型可能因为带宽、隐私或者等等原因导致无法获取到源数据
> 模型闭源/或者由于涉及隐私无法公开，如推荐系统
- 在测试过程中重新处理元数据在计算上过大导致不可行，或者不划算
> 如微调diffusiondrive v2 可能重新训练的代价太大
- 如果不进行适应，可能导致模型不够准确而无法完成其目标
> 在自动驾驶环境下，如果不微调极易发生碰撞
- TTT要求在源域训练阶段修改模型的结构，来联合优化监督损失，在缺少源数据情况下不可能

tent是一个无元数据 无目标标签 无训练过程干预 的无监督学习方法

传统tent由于关注香农熵最小化，要求训练过程的损失函数是标准交叉熵损失，只能运用于分类问题
<img width="664" height="327" alt="image" src="https://github.com/user-attachments/assets/eb2b0611-6886-4166-9c4f-698d57a1abdf" />
> TTT TTA 和fine tuning(使用带标签数据进行有监督学习)数学上的区别


## 2. Motivation

- 对于所有标准交叉熵训练的模型，都能直接套用tta直接增强
- 对缺乏源数据的情况也可以适用
- 在训练时就可以增强，节省时间和计算资源

## 3. Method
### 数学过程

1. 归一化：对模型输入数据进行调整，使其满足标准分布：

$\bar{x} = \frac{x - \mu}{\sigma}$
> 方差 $\sigma_{batch}^2$

2. 变换并迭代：利用仿射参数（ $\gamma$ 和 $\beta$）将 $\bar{x}$ 转换为输出：

$x' = \gamma\bar{x} + \beta$
>首个 $\gamma$ 和 $\beta$ 来自训练的数据

3. 反向传播 最小化平均熵公式：

$$H(\hat{y}) = - \sum_{c} p(\hat{y}_c) \log p(\hat{y}_c)$$

<img width="628" height="85" alt="image" src="https://github.com/user-attachments/assets/d7007101-d388-4978-b976-862139f6ef0b" />

## 4. Experiments

1. 在 CIFAR-10/CIFAR-100 和 ImageNet 上评估 tent 的图像损坏鲁棒性

Tent 在使用更少数据和计算量的情况下提升更大：表2 

Tent 在所有损坏类型上都取得了持续的改进：图5

Tent 在不改变训练过程的前提下达到了新的 SOTA

2. 从 SVHN 到 MNIST/MNIST-M/USPS 的数字自适应进行基准测试

Tent 在没有源数据的情况下适应目标域

Tent 需要的计算量更少，但仍然随着计算逐渐提升

Tent 可扩展至语义分割和 VisDA-C

3. 分析 
Tent 降低了熵和误差：图6

Tent 跨目标数据泛化： 自适应并不仅仅局限于用于更新的数据点。我们在目标域训练集上进行自适应，然后在目标域测试集上进行评估，误差同样下降了。

Tent 调制不同于归一化：图7

Tent 适应替代架构： Tent 在原理上是与架构无关的。tent在自注意力网络和均衡求解网络上同样降低了误差。

### 消融实验： 

不更新归一化会增加误差；不更新变换参数则该方法退化为测试时归一化。

仅仅更新模型的最后一层可以带来改善，但随着进一步优化性能会下降。

更新全部模型参数 $\theta$ 的效果，甚至不能超过未自适应的源模型。

