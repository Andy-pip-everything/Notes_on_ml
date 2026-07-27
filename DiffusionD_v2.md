# Diffusion Drive v2

<img width="785" height="434" alt="pipeline overview" src="https://github.com/user-attachments/assets/456aedf5-ac79-4f71-95c5-a847e4be6eb5" />

## 目录

**方法流程**

1. 构建 anchor 分布
2. 从截断开始前向加噪
3. 马尔可夫过程去噪
4. 乘法加噪
5. anchor 内 GRPO
6. anchor 截断修正
7. 模式选择器

**工程部分**

- 环境配置
- 复现结果

---

## 方法

### 1. 构建 anchor 分布

anchor 来自对专家 trajectory 的 k-means 聚类。

**单个 anchor 下的路径分布**

$$p(\tau^k \mid \mathbf{a}^k, z) = \mathcal{N}\big(\tau^k \mid \mathbf{a}^k + \mu^k(z),\ \Sigma^k(z)\big)$$

| 符号 | 含义 |
| --- | --- |
| $z$ | 上下文特征 |
| $\mu^k(z)$ | 均值偏移量 |
| $\Sigma^k(z)$ | 协方差矩阵，代表不确定性 |

**全局化**

$$p(\tau \mid z) = \sum_{k=1}^{N_{anchor}} s(\mathbf{a}^k \mid z)\, p(\tau^k \mid \mathbf{a}^k, z)$$

- $s(\mathbf{a}^k \mid z)$：选择该 anchor 的概率

### 2. 从截断开始前向加噪

**Diffusion 加噪**

$$\tau_t^k = \sqrt{\bar{\alpha}_t}\,\mathbf{a}^k + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

- $\sqrt{\bar{\alpha}_t}\,\mathbf{a}^k$：缩放后的原始信号
- $\bar{\alpha}_t$：噪声系数

**损失函数**

$$\mathcal{L} = \sum_{k=1}^{N_{anchor}} \Big[ y^k \mathcal{L}_{rec}(\hat{\tau}^k, \tau_{gt}) + \mathcal{L}_{BCE}(\hat{s}^k, y^k) \Big]$$

- $y^k$：真实意图
- $\mathcal{L}_{rec}$：重建损失
- $\mathcal{L}_{BCE}$：二元交叉熵损失

### 3. 马尔可夫过程去噪

**策略化**

$$\pi_\theta(\tau_{t-1}^k \mid \tau_t^k, z, \mathbf{a}^k) = \mathcal{N}\big(\tau_{t-1}^k;\ \mu_\theta(\tau_t^k, t, z, \mathbf{a}^k),\ \eta(1 - \alpha_t)\mathbf{I}\big)$$

- $\mu_\theta(\tau_t^k, t, z, \mathbf{a}^k)$：决定去噪后轨迹最可能的位置
- $\eta(1 - \alpha_t)\mathbf{I}$：高斯分布的方差，代表当前去噪步骤的随机性和不确定性

**REINFORCE 计算梯度**

$$\nabla_\theta \mathcal{J}(\pi_\theta^k) = \mathbb{E}_{\pi_\theta^k} \left[ \sum_{t=1}^{T_{trunc}} \nabla_\theta \log \pi_\theta^k(\tau_{t-1}^k \mid \tau_t^k)\, A_t^k \right]$$

| 符号 | 含义 |
| --- | --- |
| $\mathcal{J}(\pi_\theta^k)$ | 目标函数 |
| $\mathbb{E}_{\pi_\theta^k}$ | 在策略 $\pi_\theta^k$ 下取期望 |
| $\log \pi_\theta^k(\tau_{t-1}^k \mid \tau_t^k)$ | "信心"，把概率的导数转成可计算的期望形式 |
| $A_t^k$ | 优势函数；$A_t^k > 0$ 时沿正方向更新，提高该去噪动作的概率 |

### 4. 乘法加噪

$$\tau' = (1 + \epsilon_{mul})\,\tau, \qquad \epsilon_{mul} = (\epsilon_{long},\ \epsilon_{lat})$$

> 近端与远端尺度不一致，加法会破坏结构完整性；乘法自带缩放。

### 5. anchor 内 GRPO

**计算组内轨迹优势**

$$A^{k,i} = \frac{r^{k,i} - \text{mean}(\{r^{k,1}, r^{k,2}, \cdots, r^{k,G}\})}{\text{std}(\{r^{k,1}, r^{k,2}, \cdots, r^{k,G}\})}$$

衡量第 $k$ 个 anchor 组内第 $i$ 条轨迹的相对表现。

- $r^{k,i}$：轨迹的评分
- $G$：组内轨迹数量
- $\text{std}$：标准差

**RL 损失函数**

$$L_{RL} = -\frac{1}{N_{anchor}} \sum_{k=1}^{N_{anchor}} \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_{trunc}} \sum_{t=1}^{T_{trunc}} \gamma_{t-1} \log \pi_\theta(\tau_{t-1}^{k,i} \mid \tau_t^{k,i})\, A^{k,i}$$

- 三重平均：先对去噪步取均值，再对组内轨迹取均值，最后对 anchor 取均值
- $\gamma_{t-1}$：各去噪步的权重系数
- $A^{k,i}$ 在同一条轨迹的所有时间步上共享（组内相对优势，不逐步估计）

### 6. anchor 截断修正

$$A_{trunc}^{k,i} = \begin{cases} -1 & \text{if collision}, \\ \max(0,\ A^{k,i}) & \text{otherwise}. \end{cases}$$

**混合目标函数**

$$L = L_{RL} + \lambda L_{IL}$$

- $\lambda \in (0, 1)$：权重系数

### 7. 模式选择器

$$\mathcal{L}_{rank} = \frac{1}{N} \sum_{i,j} \max\big(0,\ -\text{sign}(s_i - s_j) \cdot (\hat{s}_i - \hat{s}_j) + m\big)$$

用真实质量分数之差的符号，乘以两条轨迹预测分数之差，再加上边距 $m$。

---

## 环境配置

**线程数**

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

**路径变量**

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/autodl-tmp/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/root/autodl-tmp/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/root/autodl-tmp/navsim_workspace/DiffusionDriveV2"
export OPENSCENE_DATA_ROOT="/root/autodl-tmp/navsim_workspace/dataset"
```

**目录结构**

```text
navsim_workspace/
├── DiffusionDriveV2/              # = $NAVSIM_DEVKIT_ROOT
│   ├── ckpts/                     # 权重
│   ├── gtrs_traj/
│   │   └── navtrain_16384.pkl
│   └── download/
├── exp/                           # = $NAVSIM_EXP_ROOT
│   ├── training_cache/            #  94 G
│   ├── metric_cache/              # 3.0 G   (navtest)
│   ├── train_pdm_cache/           #  33 G   (navtrain)
│   └── metric_feature_cache/      # ~11 G   (navtest)
└── dataset/                       # = $OPENSCENE_DATA_ROOT
    ├── maps/
    ├── navsim_logs/{trainval,test}/
    └── sensor_blobs/{trainval,test}/
```

---

## 复现结果


$$\text{PDMS} = \text{NC} \times \text{DAC} \times \frac{5 \cdot \text{EP} + 5 \cdot \text{TTC} + 2 \cdot \text{C}}{12}$$

| 缩写 | 指标 | 全称 |
| --- | --- | --- |
| NC | `no_collision` | No at-fault Collision |
| DAC | `drivable_area` | Drivable Area Compliance |
| DDC | `dir_weighted` | Driving Direction Compliance |
| EP | `progress` | Ego Progress |
| TTC | `ttc` | Time-to-Collision |
| C | `comfort` | Comfort |
| PDMS | `final` | 聚合总分 |

> PDMS 按逐样本计算后再求平均，因此无法用下表各列的均值反推出 `final`（乘法项存在相关性）。

### fp16 (AMP)


| 指标 | coarse | fine_0 | fine_1 | **fine_2** |
| --- | --- | --- | --- | --- |
| NC | 98.42 | 98.30 | 98.26 | **98.31** |
| DAC | 97.70 | 97.71 | 97.79 | **97.77** |
| DDC | 96.65 | 96.55 | 96.61 | **96.65** |
| EP | 85.72 | 87.34 | 87.40 | **87.34** |
| TTC | 94.81 | 94.51 | 94.45 | **94.66** |
| C | 99.60 | 99.88 | 99.89 | **99.88** |
| **PDMS** | **90.29** | **90.86** | **90.92** | **90.98** |

### fp32

| 指标 | coarse | fine_0 | fine_1 | **fine_2** |
| --- | --- | --- | --- | --- |
| NC | 98.39 | 98.21 | 98.26 | **98.37** |
| DAC | 97.70 | 97.72 | 97.74 | **97.76** |
| DDC | 96.71 | 96.53 | 96.57 | **96.65** |
| EP | 85.71 | 87.31 | 87.41 | **87.44** |
| TTC | 94.76 | 94.51 | 94.59 | **94.74** |
| C | 99.60 | 99.90 | 99.88 | **99.88** |
| **PDMS** | **90.26** | **90.85** | **90.95** | **91.06** |


| 指标 | DiffusionDrive | fp16 | fp32 |
| --- | --- | --- | --- |
| NC | 98.2 | 98.31 | 98.37 |
| DAC | 96.2 | 97.77 | 97.76 |
| TTC | 94.7 | 94.66 | 94.74 |
| EP | 82.2 | 87.34 | 87.44 |
| C | 100 | 99.88 | 99.88 |
| **PDMS** | **91.2** | **90.98** | **91.06** |

主要增益来自 EP（+5.2）和 DAC（+1.6），PDMS 与原文基本持平。

### Comparison

| 阶段 | fp16 (AMP) | fp32 | Δ (fp32 − fp16) |
| --- | --- | --- | --- |
| coarse | 90.288 | 90.255 | −0.033 |
| fine_0 | 90.861 | 90.846 | −0.015 |
| fine_1 | 90.916 | 90.946 | +0.030 |
| fine_2 | 90.981 | 91.060 | +0.079 |

