<img width="785" height="434" alt="image" src="https://github.com/user-attachments/assets/456aedf5-ac79-4f71-95c5-a847e4be6eb5" />

**构建anchor分布**

**从截断开始前向加噪**

**马尔可夫过程去噪**

**乘法加噪**

**anchor内grpo**

**anchor截断修正**

**模式选择器**



## 构建anchor分布:

 anchor来自对专家trajectory的kmeans
  
单个anchor下路径分布：
        
 $$p(\tau^k|\mathbf{a}^k,z)=\mathcal{N}(\tau^k|\mathbf{a}^k+\mu^k(z),\Sigma^k(z))$$

 z上下文特征，μ^k(z)均值偏移量，Σ^k(z)协方差矩阵（不确定性）

全局化：

 $$p(\tau|z)=\sum_{k=1}^{N_{anchor}}s(\mathbf{a}^k|z)p(\tau^k|\mathbf{a}^k,z)$$

 s(a^k|z)选择该anchor的概率
 
## 从截断开始前向加噪：

Diffusion 加噪：

$$\tau_t^k=\sqrt{\bar{\alpha}_t}\mathbf{a}^k+\sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon\sim\mathcal{N}(0,\mathbf{I})$$

$\sqrt{\bar{\alpha}_t}\mathbf{a}^k$ 缩放后原始信号， $\bar{\alpha}_t$ 噪声系数

损失函数：

$$\mathcal{L}=\sum_{k=1}^{N_{anchor}}[y^k\mathcal{L}_{rec}(\hat{\tau}^k,\tau_{gt})+\mathcal{L}_{BCE}(\hat{s}^k,y^k)]$$

y^k 真实意图， Lrec 重建损失， Lbce 二元交叉熵损失

## 马尔可夫过程去噪：

策略化：

$$\pi_\theta(\tau_{t-1}^k|\tau_t^k,z,\mathbf{a}^k)=\mathcal{N}(\tau_{t-1}^k;\mu_\theta(\tau_t^k,t,z,\mathbf{a}^k),\eta(1-\alpha_t)\mathbf{I})$$

$\mu_\theta(\tau_t^k,t,z,\mathbf{a}^k)$ 决定去噪后轨迹最可能的位置， $\eta(1-\alpha_t)\mathbf{I}$：高斯分布的方差 代表当前去噪步骤的随机性和不确定性

reinforce计算梯度：

$$\nabla_\theta\mathcal{J}(\pi_\theta^k)=\mathbb{E}_{\pi_\theta^k}\left[\sum_{t=1}^{T_{trunc}}\nabla_\theta\log\pi_\theta^k(\tau_{t-1}^k|\tau_t^k)A_t^k\right]$$

$\mathcal{J}(\pi_\theta^k)$ ： 目标函数  Ek：数学期望 

$\log\pi_\theta^k(\tau_{t-1}^k|\tau_t^k)$：信心，将概率的导数转化为可以计算的期望形式 $A_t^k$：优势函数， $A_t^k > 0$ 沿正方向增加，提高这种去噪动作的概率

## 乘法加噪

$$\tau^{\prime}=(1+\epsilon_{mul})\tau$$ ， $\epsilon_{mul}=(\epsilon_{long},\epsilon_{lat})$。

近端远端尺度不一致，加法破坏了结构完整性。乘法自带缩放。

## anchor内grpo

计算组内轨迹优势：

$$A^{k,i}=\frac{r^{k,i}-\text{mean}(\{r^{k,1},r^{k,2},\cdots,r^{k,G}\})}{\text{std}(\{r^{k,1},r^{k,2},\cdots,r^{k,G}\})}$$

计算第k个anchor组内第i条轨迹的相对表现

$r^{k,i}$： 轨迹的评分 G 轨迹数量 std 标准差

Rl损失函数：

$$L_{RL}=-\frac{1}{N_{anchor}}\sum_{k=1}^{N_{anchor}}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{trunc}}\sum_{t=1}^{T_{trunc}}\gamma_{t-1}\log\pi_\theta(\tau_{t-1}^{k,i}|\tau_t^{k,i})A^{k,i}$$

$\log\pi_\theta^k(\tau_{t-1}^k|\tau_t^k)$：信心，将概率的导数转化为可以计算的期望形式

## anchor截断修正

$$A_{trunc}^{k,i}=\begin{cases}-1 & \text{if collision}, \\ \max(0,A^{k,i}) & \text{otherwise}.\end{cases}$$

混合目标函数：

$$L=L_{RL}+\lambda L_{IL}$$ with weight coefficient λ ∈ (0,1)


## 模式选择器

$$\mathcal{L}_{rank}=\frac{1}{N}\sum_{i,j}\max(0,-\text{sign}(s_i-s_j)\cdot(\hat{s}_i-\hat{s}_j)+m)$$

真实质量分数判断符号 乘以这两条轨迹预测的分数差 加上边距
