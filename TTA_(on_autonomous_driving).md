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

# Centaur:RobustEnd-to-EndAutonomousDrivingwithTest-TimeTraining

## 1. Problem

- TENT 只能处理标准交叉熵（分类问题），而传统的端到端自动驾驶模型直接回归出一条连续的轨迹

- 像Hydra-MDP，盲目最小化全局分布的熵，模型会倾向于避开那些拥有大量相似邻居的安全轨迹

- fallback layer等预编程的规则或代价函数无法利用新的训练数据进行学习和改进

- 传统的算法依赖设计的代价函数在优化轨迹，tta或类似算法可以自学习/自优化

- TTT对自驾来说延迟太高了，需要更好的优化

<img width="728" height="292" alt="image" src="https://github.com/user-attachments/assets/5e4eba2b-6e82-4f91-973e-3e98a77adb05" />

## 2. Motivation

- 解决了tta不能用于回归问题，带来全新的可自训练的 TTT算法

- 无监督学习，不再需要大量的人工标注了

- 可接受的延迟范围

## 3. Method

### 轨迹打分：

对真实数据跑kmeans聚类，求出k=8192类中心（来自Hydra-MDP）

$$ \{s_{j}\}_{j=1}^{k}=\{\pi_{\theta}(t_{j},x_{0},c_{0})\}_{j=1}^{k} （1）$$ 

x0​,c0​ 当前帧的传感器输入、导航指令

$$ \hat{t}_{0}=arg~max\{\phi(s_{j})\}_{j=1}^{k} （2）$$

对轨迹打分，分数取NC（不碰撞）、DAC（不越界）、EP（前进效率）、C（舒适度）、TTC（碰撞时间）中最高的值，来自navsim评分规则。

### 最小化损失函数：

$$ arg~min_{\theta}\mathbb{E}_{(t,x,c)\sim D_{kd}}[\mathcal{L}_{kd}(e(t,x,c),\pi_{\theta}(t,x,c))] (3) $$

e是一个专家算法，对任意轨迹给出分数。Dkd是拿这个专家算法在模拟器里得出的知识蒸馏数据集，Lkd交叉熵

> 分布不均，不能直接对kmeans得出的轨迹计算熵

### 聚类熵 Cluster Entropy:

从这里开始不需要设计的算法

首先对k=8192进行采样，优先选择其中专家算法得分高的轨迹，采样出M条轨迹

再从M条轨迹中，按照横向偏移量选出5个锚点

按L2距离为M条轨迹各自找到最近的锚点，从而构造出5个簇。在每个簇内，我们把所有轨迹的预测分数相加，得到锚点分数，再进行归一化

$$
\left(\frac{score_{a}}{\sum_{a=1}^{5}score_{a}}\right)
$$

定义为分布的香农熵

### TTT缓冲区：

前向传播之后将求出的梯度放入缓冲区，用缓冲区的平均梯度去更新网络

$$
\hat{\theta}_{i}=\theta-\eta\{\frac{\partial H}{\partial\theta}\}_{avg}
$$

> 没有找到官方发布的代码 Hydra-MDP也没有公布


## 4. Experiment

Baseline Hydra-MDP 消融实验 M=100 buffer F=4

<img width="935" height="301" alt="image" src="https://github.com/user-attachments/assets/dc868034-3b6a-4c8f-b386-cf5d94b0785e" />

KL散度：对5个评分指标各自在 k=8192 上得到分布，两两求kL散度再求和

Full Entropy：类似Cluster Entropy，但跳过聚类，直接对 100 维分布算熵

Cluster Entropy：聚类，在欧氏轨迹空间按横向偏移分 5 个方向簇

ttt +1.2 Full Entropy +0.3 Cluster Entropy +0.8

### 创新点：

Cluster Entropy

方向聚类锚点

TTT缓冲区
