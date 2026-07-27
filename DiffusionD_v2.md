<img width="785" height="434" alt="image" src="https://github.com/user-attachments/assets/456aedf5-ac79-4f71-95c5-a847e4be6eb5" />

# Diffusion Drive v2

## 流程分布
- **构建anchor分布**
- **从截断开始前向加噪**
- **马尔可夫过程去噪**
- **乘法加噪**
- **anchor内grpo**
- **anchor截断修正**
- **模式选择器**


## 1. 构建anchor分布

anchor来自对专家trajectory的kmeans聚类。
  
### 单个anchor下路径分布：
$$p(\tau^k|\mathbf{a}^k,z)=\mathcal{N}(\tau^k|\mathbf{a}^k+\mu^k(z),\Sigma^k(z))$$

- $z$：上下文特征
- $\mu^k(z)$：均值偏移量
- $\Sigma^k(z)$：协方差矩阵（代表不确定性）

### 全局化：
$$p(\tau|z)=\sum_{k=1}^{N_{anchor}}s(\mathbf{a}^k|z)p(\tau^k|\mathbf{a}^k,z)$$

- $s(\mathbf{a}^k|z)$：选择该anchor的概率
 

## 2. 从截断开始前向加噪

### Diffusion 加噪：
$$\tau_t^k=\sqrt{\bar{\alpha}_t}\mathbf{a}^k+\sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon\sim\mathcal{N}(0,\mathbf{I})$$

- $\sqrt{\bar{\alpha}_t}\mathbf{a}^k$：缩放后原始信号
- $\bar{\alpha}_t$：噪声系数

### 损失函数：
$$\mathcal{L}=\sum_{k=1}^{N_{anchor}}[y^k\mathcal{L}_{rec}(\hat{\tau}^k,\tau_{gt})+\mathcal{L}_{BCE}(\hat{s}^k,y^k)]$$

- $y^k$：真实意图
- $\mathcal{L}_{rec}$：重建损失
- $\mathcal{L}_{BCE}$：二元交叉熵损失


## 3. 马尔可夫过程去噪

### 策略化：
$$\pi_\theta(\tau_{t-1}^k|\tau_t^k,z,\mathbf{a}^k)=\mathcal{N}(\tau_{t-1}^k;\mu_\theta(\tau_t^k,t,z,\mathbf{a}^k),\eta(1-\alpha_t)\mathbf{I})$$

- $\mu_\theta(\tau_t^k,t,z,\mathbf{a}^k)$：决定去噪后轨迹最可能的位置
- $\eta(1-\alpha_t)\mathbf{I}$：高斯分布的方差，代表当前去噪步骤的随机性和不确定性

### REINFORCE 计算梯度：
$$\nabla_\theta\mathcal{J}(\pi_\theta^k)=\mathbb{E}_{\pi_\theta^k}\left[\sum_{t=1}^{T_{trunc}}\nabla_\theta\log\pi_\theta^k(\tau_{t-1}^k|\tau_t^k)A_t^k\right]$$

- $\mathcal{J}(\pi_\theta^k)$：目标函数 
- $\mathbb{E}_k$：数学期望 
- $\log\pi_\theta^k(\tau_{t-1}^k|\tau_t^k)$：信心，将概率的导数转化为可以计算的期望形式 
- $A_t^k$：优势函数，当 $A_t^k > 0$ 时沿正方向增加，提高这种去噪动作的概率


## 4. 乘法加噪

$$\tau^{\prime}=(1+\epsilon_{mul})\tau, \quad \epsilon_{mul}=(\epsilon_{long},\epsilon_{lat})$$

> 近端远端尺度不一致，加法破坏了结构完整性。乘法自带缩放。


## 5. anchor内grpo

### 计算组内轨迹优势：
$$A^{k,i}=\frac{r^{k,i}-\text{mean}(\{r^{k,1},r^{k,2},\cdots,r^{k,G}\})}{\text{std}(\{r^{k,1},r^{k,2},\cdots,r^{k,G}\})}$$

- 计算第 $k$ 个 anchor 组内第 $i$ 条轨迹的相对表现
- $r^{k,i}$：轨迹的评分 
- $G$：轨迹数量 
- $\text{std}$：标准差

### RL损失函数：
$$L_{RL}=-\frac{1}{N_{anchor}}\sum_{k=1}^{N_{anchor}}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{trunc}}\sum_{t=1}^{T_{trunc}}\gamma_{t-1}\log\pi_\theta(\tau_{t-1}^{k,i}|\tau_t^{k,i})A^{k,i}$$

- $\log\pi_\theta^k(\tau_{t-1}^k|\tau_t^k)$：信心，将概率的导数转化为可以计算的期望形式

## 6. anchor截断修正

$$A_{trunc}^{k,i}=\begin{cases}-1 & \text{if collision}, \\ \max(0,A^{k,i}) & \text{otherwise}.\end{cases}$$

### 混合目标函数：
$$L=L_{RL}+\lambda L_{IL}$$

- 权重系数 $\lambda \in (0,1)$


## 7. 模式选择器

$$\mathcal{L}_{rank}=\frac{1}{N}\sum_{i,j}\max(0,-\text{sign}(s_i-s_j)\cdot(\hat{s}_i-\hat{s}_j)+m)$$

- 真实质量分数判断符号乘以这两条轨迹预测的分数差加上边距

> __________________

线程数量
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

环境变量
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/autodl-tmp/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/root/autodl-tmp/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/root/autodl-tmp/navsim_workspace/DiffusionDriveV2"
export OPENSCENE_DATA_ROOT="/root/autodl-tmp/navsim_workspace/dataset"

目录结构
navsim_workspace/
├── DiffusionDriveV2/          # devkit
│   ├── ckpts/                 # 权重
│   ├── gtrs_traj/navtrain_16384.pkl
│   └── download/
├── exp/                       # = $NAVSIM_EXP_ROOT
│   ├── training_cache/        #  94 G 
│   ├── metric_cache/          # 3.0 G  (navtest)
│   ├── train_pdm_cache/       #  33 G  (navtrain)
│   └── metric_feature_cache/  # ~11 G  (navtest)
└── dataset/                   # = $OPENSCENE_DATA_ROOT
    ├── maps/
    ├── navsim_logs/{trainval,test}/
    └── sensor_blobs/{trainval,test}/

## 复现结果

PDMS = NC × DAC × (5·EP + 5·TTC + 2·C) / 12

fp16

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Validate metric         ┃          DataLoader 0          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    val/coarse_comfort_epoch    │       0.9959657500411658       │
│ val/coarse_dir_weighted_epoch  │       0.9665321916680388       │
│ val/coarse_drivable_area_epoch │       0.977029474724189        │
│     val/coarse_final_epoch     │       0.9028792452285785       │
│ val/coarse_no_collision_epoch  │       0.9841511608760085       │ 
│   val/coarse_progress_epoch    │       0.8571524831711747       │
│    val/coarse_reward_epoch     │       0.9028792977333069       │
│      val/coarse_ttc_epoch      │       0.9481310719578462       │
│    val/fine_0_comfort_epoch    │       0.9987650255228059       │
│ val/fine_0_dir_weighted_epoch  │       0.9655030462703771       │
│ val/fine_0_drivable_area_epoch │       0.9771118063560019       │
│     val/fine_0_final_epoch     │       0.9086108425043932       │
│ val/fine_0_no_collision_epoch  │       0.9829573522147209       │
│   val/fine_0_progress_epoch    │       0.8734145032112113       │
│      val/fine_0_ttc_epoch      │       0.9450848015807674       │
│    val/fine_1_comfort_epoch    │       0.9989296887864317       │
│ val/fine_1_dir_weighted_epoch  │       0.9660793676930677       │
│ val/fine_1_drivable_area_epoch │       0.9779351226741314       │
│     val/fine_1_final_epoch     │       0.9091624979731457       │
│ val/fine_1_no_collision_epoch  │       0.9825868598715627       │
│   val/fine_1_progress_epoch    │       0.8740466719001095       │
│      val/fine_1_ttc_epoch      │       0.9445084801580768       │
│    val/fine_2_comfort_epoch    │       0.9988473571546188       │
│ val/fine_2_dir_weighted_epoch  │       0.9665321916680388       │ DDC
│ val/fine_2_drivable_area_epoch │       0.9776881277786926       │ Drivable Area Compliance
│     val/fine_2_final_epoch     │       0.9098061907133913       │ <--聚合总分
│ val/fine_2_no_collision_epoch  │       0.9831220154783468       │ No at-fault Collision
│   val/fine_2_progress_epoch    │       0.8733748233954506       │ Ego Progress
│      val/fine_2_ttc_epoch      │       0.9465667709534002       │ Time-to-Collision
│    val/fine_reward_0_epoch     │       0.9086114764213562       │
│    val/fine_reward_1_epoch     │       0.9091625809669495       │
│    val/fine_reward_2_epoch     │       0.9098061919212341       │
└────────────────────────────────┴────────────────────────────────┘

| 指标 | fine_2 | DiffusionDrive |
| --- | --- | --- |
| no_collision | 98.31 | 98.2 |
| drivable_area | 97.77 | 96.2 |
| ttc | 94.66 | 94.7 |
| progress | 87.34 | 82.2 |
| comfort | 99.88 | 100 |
| PDMS | 90.98 | 91.2 |

fp32

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Validate metric         ┃          DataLoader 0          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    val/coarse_comfort_epoch    │       0.9959657500411658       │
│ val/coarse_dir_weighted_epoch  │       0.967067347274823        │
│ val/coarse_drivable_area_epoch │       0.977029474724189        │
│     val/coarse_final_epoch     │       0.9025531070890828       │
│ val/coarse_no_collision_epoch  │       0.9839453317964763       │
│   val/coarse_progress_epoch    │       0.8570548772480697       │
│    val/coarse_reward_epoch     │       0.9025532603263855       │
│      val/coarse_ttc_epoch      │       0.9476370821669685       │
│    val/fine_0_comfort_epoch    │       0.9990120204182447       │
│ val/fine_0_dir_weighted_epoch  │       0.9652560513749383       │
│ val/fine_0_drivable_area_epoch │       0.977194137987815        │
│     val/fine_0_final_epoch     │       0.9084570284959671       │
│ val/fine_0_no_collision_epoch  │       0.9821340358965914       │
│   val/fine_0_progress_epoch    │       0.8731276798830557       │
│      val/fine_0_ttc_epoch      │       0.9450848015807674       │
│    val/fine_1_comfort_epoch    │       0.9987650255228059       │
│ val/fine_1_dir_weighted_epoch  │       0.9657088753499095       │
│ val/fine_1_drivable_area_epoch │       0.9773588012514408       │
│     val/fine_1_final_epoch     │       0.9094644807842343       │
│ val/fine_1_no_collision_epoch  │       0.9825868598715627       │
│   val/fine_1_progress_epoch    │       0.8741294303795697       │
│      val/fine_1_ttc_epoch      │       0.9459081178988967       │
│    val/fine_2_comfort_epoch    │       0.9988473571546188       │
│ val/fine_2_dir_weighted_epoch  │       0.9665321916680388       │
│ val/fine_2_drivable_area_epoch │       0.9776057961468796       │
│     val/fine_2_final_epoch     │       0.9106009595011908       │
│ val/fine_2_no_collision_epoch  │       0.9836983369010374       │
│   val/fine_2_progress_epoch    │       0.8744262696075553       │
│      val/fine_2_ttc_epoch      │       0.9473900872715297       │
│    val/fine_reward_0_epoch     │       0.9084572196006775       │
│    val/fine_reward_1_epoch     │       0.9094640016555786       │
│    val/fine_reward_2_epoch     │       0.910601019859314        │
└────────────────────────────────┴────────────────────────────────┘

| 指标 | fine_2 | DiffusionDrive |
| --- | --- | --- |
| no_collision | 98.37 | 98.2 |
| drivable_area | 97.76 | 96.2 |
| ttc | 94.66 | 94.74 |
| progress | 87.44 | 82.2 |
| comfort | 99.88 | 100 |
| PDMS | 91.06 | 91.2 |

Comparasion

阶段	fp16 (AMP)	fp32	变化

coarse	90.288	90.255	−0.033

fine_0	90.861	90.846	−0.015

fine_1	90.916	90.946	+0.030

fine_2	90.981	91.060	+0.079
