Readme.md just simply copied content of BridgeDrive.md!

# Hands_on_bridgedrive

## Core process
- **Diffusion Bridge Overview**
- **Planning Task Formulation**
- **Joint Distribution Factorization**
- **Mathematical Principles of Standard Diffusion**
- **Mathematical Modifications for BridgeDrive**
- **End-to-end Autonomous Driving Modalities**



## 1. Diffusion Bridge Overview

DiffusionDrive does not start with pure noise, but rather with an anchor that has had a small amount of noise added for denoising.

BridgeDrive mathematically formalizes the planning task as a Diffusion Bridge. This ensures that the forward and reverse denoising processes are perfectly symmetrical.

<img width="1446" height="604" alt="BridgeDrive_algorithm" src="https://github.com/user-attachments/assets/097d8c60-255c-4f59-8142-46c06f4f4a32" />

> BridgeDrive is compatible with efficient ODE solvers, enabling real-time deployment.



## 2. Planning Task Formulation

The planning task in autonomous driving can be formulated as predicting future trajectories of the ego-vehicle based on raw sensor inputs.

- (1) Temporal speed waypoints
- (2) Geometric path waypoints

Highly different between Open-loop and closed-loop setting. CARLA is most widely used platform.

### Forward Process:
Mathematically, the forward process can be concluded as:
$$dx_t=f(t)x_tdt+g(t)dw_t, \quad x_0\sim p_d \quad (1)$$

### Caculating the loss function:
$$\frac{dx_t}{dt} = f(t)x_t - \frac{g(t)^2}{2} \mathbf{\nabla_{x_t}\log q(x_t)}$$

$$\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} = \frac{\alpha_t x_0 - x_t}{\sigma_t^2}$$

$$\min_\theta \mathbb{E} [w(t) \lVert x_\theta(x_t,t) - x_0 \rVert^2]$$



## 3. Joint Distribution Factorization

To incorporate anchors into diffusion models in a principled way, we propose to factorize the joint distribution of the ground-truth trajectory $x$, anchor $y$, and guidance information $z$ as:

$$p_d(x,y,z) = p_d(x|y,z)p_d(y|z)p_d(z)$$

The model constructs a direct bridge between the ground-truth trajectory ($x_0 = x$) and the anchor ($x_T = y$). Because the SDE is linear, it yields an analytical Gaussian transition kernel, enabling efficient, simulation-free training:

$$q(x_t|x_0,x_T) = \mathcal{N}(x_t | a_t x_T + b_t x_0, c_t^2 I)$$

Unlike DiffusionDrive, both forward and reverse paths perfectly align at $x_0$. The denoiser is trained to reverse this exact bridge by minimizing the error against the ground truth:

$$\min_\theta \mathbb{E} \left[ w(t) \lVert x_\theta(x_t, t, x_T, z) - x_0 \rVert^2 \right]$$

To physically realize the joint distribution factorization $p_d(x,y,z) = p_d(x|y,z)p_d(y|z)p_d(z)$, the model introduces an independent Anchor Classifier $h_\phi(z, \mathcal{Y})$. This classifier predicts the most suitable anchor $y$ given the scene context $z$, effectively determining the starting point ($x_T=y$) for the diffusion bridge prior to the denoising process.



## 4. Mathematical Principles of Standard Diffusion

The standard diffusion model aims to generate data $x_0 \sim p_d(x_0)$ from pure Gaussian noise $x_T \sim \mathcal{N}(0, \sigma_{\max}^2 I)$ by mathematically reverting a forward noise-adding process.

### SDE:
$$dx_t = f(t)x_t dt + g(t)dw_t, \quad x_0 \sim p_d$$

- $f(t)$：the linear drift coefficient
- $g(t)$：the diffusion coefficient controlling noise scale
- $w_t$：standard Brownian motion

### Gaussian Transition Kernel:
Because the forward SDE is linear, transitioning from the clean data $x_0$ to any arbitrary intermediate timestep $t$ yields a closed-form Gaussian distribution:

$$q(x_t|x_0) = \mathcal{N}(x_t | \alpha_t x_0, \sigma_t^2 I)$$

- $\alpha_t = \exp\left(\int_0^t f(s)ds\right)$：represents the signal scaling
- $\sigma_t^2 = \alpha_t^2 \int_0^t \frac{g(s)^2}{\alpha_s^2} ds$：represents the accumulated noise variance

### Reverse Process (Probability Flow ODE):
To generate data from noise, we simulate a Probability Flow ODE (PF-ODE) backward in time. This ODE is proven to share the exact same marginal densities $q(x_t)$ as the forward SDE:

$$\frac{dx_t}{dt} = f(t)x_t - \frac{g(t)^2}{2}\nabla_{x_t}\log q(x_t)$$

To solve this ODE, everything is known except the score function $\nabla_{x_t}\log q(x_t)$, which points toward the highest density of true data.

### Score Matching and the Loss Function (Vincent's Trick):
Calculating the true marginal score $\nabla_{x_t}\log q(x_t)$ is intractable. However, relying on the Gaussian transition kernel, the conditional score is easily derived:

$$\nabla_{x_t}\log q(x_t|x_0) = \frac{\alpha_t x_0 - x_t}{\sigma_t^2}$$

Thus, we train a neural network denoiser $x_\theta(x_t, t)$ to predict the clean data $x_0$. By minimizing the Mean Squared Error (MSE) between the prediction and actual $x_0$:

$$\min_\theta \mathbb{E}_{p(t)p_d(x_0)q(x_t|x_0)} \left[ w(t) \lVert x_\theta(x_t,t) - x_0 \rVert^2 \right]$$

> The network implicitly learns to approximate the score function as $\nabla_{x_t}\log q(x_t) \approx \frac{\alpha_t x_\theta(x_t, t) - x_t}{\sigma_t^2}$, effectively solving the ODE and completing the generation process.



## 5. Mathematical Modifications for BridgeDrive

From Standard Diffusion to Diffusion Bridge

To build the "Diffusion Bridge" and fix the asymmetry issue of previous methods, the model applies four critical mathematical modifications to the standard diffusion framework:

### 1. Replacing the Pure Noise Endpoint
Instead of terminating the forward process at pure Gaussian noise, BridgeDrive anchors the endpoint to the coarse expert trajectory $y$:
$$x_T := y$$

### 2. Rewriting the Forward SDE
To guarantee that the state precisely reaches the anchor $y$ at time $T$, a drift term (based on Doob's h-transform) is injected into the standard SDE, formulating the conditional Diffusion Bridge SDE:
$$dx_t = \left[ f(t)x_t + g(t)^2 \nabla_{x_t} \log q(x_T|x_t) \right] dt + g(t)dw_t, \quad x_0 \sim p_d, \quad x_T = y$$

### 3. Deriving the Analytical Gaussian Transition Kernel
Since the modified SDE remains linear, the intermediate state $x_t$ maintains a closed-form Gaussian distribution. It effectively acts as a direct interpolation between the ground truth $x_0$ and the anchor $x_T$:
$$q(x_t|x_0,x_T) = \mathcal{N}(x_t | a_t x_T + b_t x_0, c_t^2 I)$$

### 4. The Symmetrical Objective Function
With the bridge established, the neural network denoiser $x_\theta$ learns to predict the pristine trajectory $x_0$, conditioned on both the anchor $x_T$ and the environment context $z$. This formulation perfectly aligns the forward and reverse processes:
$$\min_\theta \mathbb{E} \left[ w(t) \lVert x_\theta(x_t, t, x_T, z) - x_0 \rVert^2 \right]$$

### Inference Stage:
During the inference (planning) stage, the model translates the chosen anchor $x_T$ into a refined trajectory $x_0$ using a specialized Bridge Probability Flow ODE (PF-ODE):
$$\frac{dx_t}{dt}=f(t)x_t-g(t)^2 \left( \frac{\nabla_{x_t}\log q(x_t|x_T,z)}{2}-\nabla_{x_t}\log q(x_T|x_t) \right)$$

> The trained denoiser $x_\theta(x_t, t, x_T, z)$ is utilized to approximate the score function. This score guides a numerical ODE solver (e.g., DDIM) to iteratively update the trajectory, efficiently bridging the coarse anchor to the precise final plan. 



## 6. End-to-end Autonomous Driving Modalities

- **End-to-end autonomous driving**: map raw sensory inputs directly to trajectory predictions or control commands.
- **Deterministic planners**: fusing multi modal sensor inputs through transformer-based encoders and decoding them into trajectory outputs via compact MLP heads. These models highlight the importance of effective sensor fusion in improving closed-loop driving performance.
- **Diffusion-based planners**: utilizing to model the multi-modal nature of human driving behaviors, generating diverse and feasible trajectories from random noise through an iterative denoising process.
- **Patterns collapse**: When patterns collapse, different random noise inputs tend to converge to similar trajectories during the denoising process.



> **Out-of distribution failure scenario**: despite BridgeDrive's exceptional modeling capabilities, it cannot generalize to out-of-distribution (OOD) scenarios.


# Hands_on_bridgedrive

## 核心流程分布
- **概述**
- **规划任务建模**
- **联合分布因式分解**
- **标准 Diffusion 的数学原理**
- **BridgeDrive 的数学重构**
- **端到端自动驾驶模式**



## 1. 概述

DiffusionDrive 并非从纯噪声开始，而是从一个添加了少量噪声的 anchor 开始进行去噪。

BridgeDrive 在数学上将规划任务形式化为 Diffusion Bridge。这确保了前向加噪和反向去噪过程是完全对称的。

<img width="1446" height="604" alt="BridgeDrive_algorithm" src="https://github.com/user-attachments/assets/097d8c60-255c-4f59-8142-46c06f4f4a32" />

> BridgeDrive 兼容高效的 ODE 求解器，能够实现实时部署。



## 2. 规划任务建模

自动驾驶中的规划任务可以建模为：基于原始传感器输入预测自车未来的轨迹。

- (1) 时域速度路点
- (2) 几何路径路点

开环与闭环设置之间存在显著差异。CARLA 是目前最广泛使用的平台。

### 前向过程：
在数学上，前向过程可以归纳为：
$$dx_t=f(t)x_tdt+g(t)dw_t, \quad x_0\sim p_d \quad (1)$$

### 损失函数计算：
$$\frac{dx_t}{dt} = f(t)x_t - \frac{g(t)^2}{2} \mathbf{\nabla_{x_t}\log q(x_t)}$$

$$\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} = \frac{\alpha_t x_0 - x_t}{\sigma_t^2}$$

$$\min_\theta \mathbb{E} [w(t) \lVert x_\theta(x_t,t) - x_0 \rVert^2]$$



## 3. 联合分布因式分解

为了以原则性的方式将 anchor 引入 diffusion 模型，我们提出将真实轨迹 $x$、anchor $y$ 和引导信息 $z$ 的联合分布因式分解为：

$$p_d(x,y,z) = p_d(x|y,z)p_d(y|z)p_d(z)$$

该模型在真实轨迹（$x_0 = x$）和 anchor（$x_T = y$）之间构建了一座直接的桥梁。由于该 SDE（随机微分方程）是线性的，它产生了一个解析的高斯转移核，从而实现了高效、无需仿真的训练：

$$q(x_t|x_0,x_T) = \mathcal{N}(x_t | a_t x_T + b_t x_0, c_t^2 I)$$

与 DiffusionDrive 不同，前向和反向路径在 $x_0$ 处完美对齐。通过最小化与真实值之间的误差，去噪器被训练用于精确逆转这一桥接过程：

$$\min_\theta \mathbb{E} \left[ w(t) \lVert x_\theta(x_t, t, x_T, z) - x_0 \rVert^2 \right]$$

为了从物理上实现联合分布因式分解 $p_d(x,y,z) = p_d(x|y,z)p_d(y|z)p_d(z)$，模型引入了一个独立的 Anchor 分类器 $h_\phi(z, \mathcal{Y})$。该分类器在给定场景上下文 $z$ 的情况下预测最合适的 anchor $y$，从而在去噪过程开始前有效地确定了 diffusion bridge 的起点（$x_T=y$）。



## 4. 标准 Diffusion 的数学原理

标准的 diffusion 模型旨在通过数学上逆转前向加噪过程，从纯高斯噪声 $x_T \sim \mathcal{N}(0, \sigma_{\max}^2 I)$ 中生成数据 $x_0 \sim p_d(x_0)$。

### SDE（随机微分方程）：
$$dx_t = f(t)x_t dt + g(t)dw_t, \quad x_0 \sim p_d$$

- $f(t)$：线性漂移系数
- $g(t)$：控制噪声尺度的扩散系数
- $w_t$：标准布朗运动

### 高斯转移核：
由于前向 SDE 是线性的，从干净数据 $x_0$ 转移到任意中间时间步 $t$ 都会产生一个闭式的高斯分布：

$$q(x_t|x_0) = \mathcal{N}(x_t | \alpha_t x_0, \sigma_t^2 I)$$

- $\alpha_t = \exp\left(\int_0^t f(s)ds\right)$：代表信号缩放
- $\sigma_t^2 = \alpha_t^2 \int_0^t \frac{g(s)^2}{\alpha_s^2} ds$：代表累积的噪声方差

### 反向过程（概率流 ODE）：
为了从噪声中生成数据，我们在时间上反向模拟概率流 ODE（PF-ODE）。已证明该 ODE 与前向 SDE 共享完全相同的边缘密度 $q(x_t)$：

$$\frac{dx_t}{dt} = f(t)x_t - \frac{g(t)^2}{2}\nabla_{x_t}\log q(x_t)$$

为了求解该 ODE，除分数函数（Score Function） $\nabla_{x_t}\log q(x_t)$ 外的所有项均为已知，该函数指向真实数据密度的最高方向。

### 分数匹配与损失函数（Vincent 技巧）：
计算真实的边缘分数 $\nabla_{x_t}\log q(x_t)$ 是难以处理的。然而，依赖于高斯转移核，可以轻松推导出条件分数：

$$\nabla_{x_t}\log q(x_t|x_0) = \frac{\alpha_t x_0 - x_t}{\sigma_t^2}$$

因此，我们训练一个神经网络去噪器 $x_\theta(x_t, t)$ 来预测干净数据 $x_0$。通过最小化预测值与实际 $x_0$ 之间的均方误差（MSE）：

$$\min_\theta \mathbb{E}_{p(t)p_d(x_0)q(x_t|x_0)} \left[ w(t) \lVert x_\theta(x_t,t) - x_0 \rVert^2 \right]$$

> 网络隐式地学习将分数函数近似为 $\nabla_{x_t}\log q(x_t) \approx \frac{\alpha_t x_\theta(x_t, t) - x_t}{\sigma_t^2}$，从而有效求解 ODE 并完成生成过程。



## 5. BridgeDrive 的数学重构

从标准 Diffusion 到 Diffusion Bridge

为了构建“Diffusion Bridge”并解决以往方法的不对称问题，该模型对标准 diffusion 框架应用了四项关键的数学重构：

### 1. 替换纯噪声终点
BridgeDrive 并非将前向过程终止于纯高斯噪声，而是将终点锚定在粗略的专家轨迹 $y$ 上：
$$x_T := y$$

### 2. 重写前向 SDE
为了保证状态在时间 $T$ 精确到达 anchor $y$，在标准 SDE 中注入了一个漂移项（基于 Doob 的 h-变换），从而形成了条件 Diffusion Bridge SDE：
$$dx_t = \left[ f(t)x_t + g(t)^2 \nabla_{x_t} \log q(x_T|x_t) \right] dt + g(t)dw_t, \quad x_0 \sim p_d, \quad x_T = y$$

### 3. 推导解析高斯转移核
由于修改后的 SDE 仍然是线性的，中间状态 $x_t$ 保持了闭式的高斯分布。它有效地充当了真实值 $x_0$ 与 anchor $x_T$ 之间的直接插值：
$$q(x_t|x_0,x_T) = \mathcal{N}(x_t | a_t x_T + b_t x_0, c_t^2 I)$$

### 4. 对称目标函数
建立桥接后，神经网络去噪器 $x_\theta$ 学习预测原始轨迹 $x_0$，其条件为 anchor $x_T$ 和环境上下文 $z$。这种公式化表达完美对齐了前向和反向过程：
$$\min_\theta \mathbb{E} \left[ w(t) \lVert x_\theta(x_t, t, x_T, z) - x_0 \rVert^2 \right]$$

### 推理阶段：
在推理（规划）阶段，模型使用专用的桥接概率流 ODE（PF-ODE），将选定的 anchor $x_T$ 转化为精确的轨迹 $x_0$：
$$\frac{dx_t}{dt}=f(t)x_t-g(t)^2 \left( \frac{\nabla_{x_t}\log q(x_t|x_T,z)}{2}-\nabla_{x_t}\log q(x_T|x_t) \right)$$

> 训练好的去噪器 $x_\theta(x_t, t, x_T, z)$ 用于近似分数函数。该分数引导数值 ODE 求解器（如 DDIM）迭代更新轨迹，高效地将粗略的 anchor 桥接到精确的最终规划结果。



## 6. 端到端自动驾驶模式

- **端到端自动驾驶 (End-to-end autonomous driving)**：将原始传感器输入直接映射为轨迹预测或控制指令。
- **确定性规划器 (Deterministic planners)**：通过基于 Transformer 的编码器融合多模态传感器输入，并通过紧凑的 MLP 头将其解码为轨迹输出。这些模型突出了有效的传感器融合在改善闭环驾驶性能中的重要性。
- **基于 Diffusion 的规划器 (Diffusion-based planners)**：用于对人类驾驶行为的多模态特性进行建模，通过迭代去噪过程从随机噪声中生成多样化且可行的轨迹。
- **模式崩溃 (Patterns collapse)**：当发生模式崩溃时，不同的随机噪声输入在去噪过程中倾向于收敛为相似的轨迹。



> **分布外 (OOD) 失败场景**：尽管 BridgeDrive 具备出色的建模能力，但它无法泛化到分布外（OOD）场景。
