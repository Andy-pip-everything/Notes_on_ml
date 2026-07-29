## DeepSeekMath: Pushing the Limits of MathematicalReasoning in Open Language Models

## 1. Problem

- ppo评价器巨量开销，训练模型资源需求量极大
- critic训练效果不够精准，预测结果带有偏差

## 2. Motivation

- 去掉critic，释放大量算力和资源占用
- 精准无偏的baseline拟合
- 准确的advantage

## 3. Method

<img width="876" height="379" alt="image" src="https://github.com/user-attachments/assets/f291a924-98f7-4db4-8247-e8fafdff6124" />

初始化参数，备份现有模型

进行M次循环：抽出问题，冻结模型生成G个问题，计算出分数

计算相对优势，更新模型参数和奖励模型

<img width="770" height="64" alt="image" src="https://github.com/user-attachments/assets/880f43dd-a5b3-454a-b9e7-84abc77e4c1e" />

<img width="789" height="98" alt="image" src="https://github.com/user-attachments/assets/5178a82f-3c9c-4e22-a3b1-5a9b3a9cb310" />

与PPO相比，修改了：

| 对比维度 | PPO | GRPO |
| :--- | :--- | :--- |
| **采样 (Sampling)** | 单条输出 $o$ | 一组输出 $\{o_i\}_{i=1}^G$ |
| **优势估算 (Advantage)** | 使用 GAE + 价值网络 $V_\psi$ | 使用组内归一化奖励 |
| **KL 散度 (KL Penalty)** | 混入 Reward 中（通常作为 per-token 惩罚，见式2） | 直接作为 Loss 的一项 |

## 4. Experiment
<img width="858" height="747" alt="image" src="https://github.com/user-attachments/assets/808bf908-0d18-4a8c-be15-6f8278251810" />
