Hands_on_bridgedrive

## 1

DiffusionDrive does not start with pure noise, but rather with an anchor that has had a small amount of noise added for denoising.

BridgeDrive mathematically formalizes the planning task as a Diffusion Bridge. This ensures that the forward and reverse denoising processes are perfectly symmetrical.

<img width="1446" height="604" alt="BridgeDrive_algorithm" src="https://github.com/user-attachments/assets/097d8c60-255c-4f59-8142-46c06f4f4a32" />

BridgeDrive is compatible with efficient ODE solvers, enabling real-time deployment.

<br>

## 2

The planning task in autonomous driving can be formulated as predicting future trajectories of the ego-vehicle based on raw sensor inputs.

(1) Temporal speed waypoints

(2) Geometric path waypoints

Highly different between Open-loop and closed-loop setting. CARLA is most widely used platform.

Mathematically, the forward process can be concluded as:

$$dx_t=f(t)x_tdt+g(t)dw_t,x_0\sim p_d,(1)$$

Caculating the loss function:

$$\frac{dx_t}{dt} = f(t)x_t - \frac{g(t)^2}{2} \mathbf{\nabla_{x_t}\log q(x_t)}$$

$$\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} = \frac{\alpha_t x_0 - x_t}{\sigma_t^2}$$

$$\min_\theta \mathbb{E} [w(t) \lVert x_\theta(x_t,t) - x_0 \rVert^2]$$

## 3

To incorporate anchors into diffusion models in a principled way, we propose to factorize the joint distribution of the ground-truth trajectory x, anchor y, and guidance information z as

$$p_d(x,y,z) = p_d(x|y,z)p_d(y|z)p_d(z)$$

The model constructs a direct bridge between the ground-truth trajectory ($x_0 = x$) and the anchor ($x_T = y$). Because the SDE is linear, it yields an analytical Gaussian transition kernel, enabling efficient, simulation-free training:

$$q(x_t|x_0,x_T) = \mathcal{N}(x_t | a_t x_T + b_t x_0, c_t^2 I)$$

Unlike DiffusionDrive, both forward and reverse paths perfectly align at $x_0$. The denoiser is trained to reverse this exact bridge by minimizing the error against the ground truth:

$$\min_\theta \mathbb{E} \left[ w(t) \lVert x_\theta(x_t, t, x_T, z) - x_0 \rVert^2 \right]$$

To physically realize the joint distribution factorization $p_d(x,y,z) = p_d(x|y,z)p_d(y|z)p_d(z)$, the model introduces an independent Anchor Classifier $h_\phi(z, \mathcal{Y})$. This classifier predicts the most suitable anchor $y$ given the scene context $z$, effectively determining the starting point ($x_T=y$) for the diffusion bridge prior to the denoising process.

## 4

Mathematical Principles of Standard Diffusion:

The standard diffusion model aims to generate data $x_0 \sim p_d(x_0)$ from pure Gaussian noise $x_T \sim \mathcal{N}(0, \sigma_{\max}^2 I)$ by mathematically reverting a forward noise-adding process.

SDE:


$$dx_t = f(t)x_t dt + g(t)dw_t, \quad x_0 \sim p_d$$


where $f(t)$ is the linear drift coefficient, $g(t)$ is the diffusion coefficient controlling noise scale, and $w_t$ is standard Brownian motion.

Gaussian Transition Kernel:

Because the forward SDE is linear, transitioning from the clean data $x_0$ to any arbitrary intermediate timestep $t$ yields a closed-form Gaussian distribution:


$$q(x_t|x_0) = \mathcal{N}(x_t | \alpha_t x_0, \sigma_t^2 I)$$


where $\alpha_t = \exp\left(\int_0^t f(s)ds\right)$ represents the signal scaling, and $\sigma_t^2 = \alpha_t^2 \int_0^t \frac{g(s)^2}{\alpha_s^2} ds$ represents the accumulated noise variance.

Reverse Process (Probability Flow ODE):

To generate data from noise, we simulate a Probability Flow ODE (PF-ODE) backward in time. This ODE is proven to share the exact same marginal densities $q(x_t)$ as the forward SDE:


$$\frac{dx_t}{dt} = f(t)x_t - \frac{g(t)^2}{2}\nabla_{x_t}\log q(x_t)$$


To solve this ODE, everything is known except the score function $\nabla_{x_t}\log q(x_t)$, which points toward the highest density of true data.

Score Matching and the Loss Function (Vincent's Trick):
Calculating the true marginal score $\nabla_{x_t}\log q(x_t)$ is intractable. However, relying on the Gaussian transition kernel, the conditional score is easily derived:


$$\nabla_{x_t}\log q(x_t|x_0) = \frac{\alpha_t x_0 - x_t}{\sigma_t^2}$$


Thus, we train a neural network denoiser $x_\theta(x_t, t)$ to predict the clean data $x_0$. By minimizing the Mean Squared Error (MSE) between the prediction and actual $x_0$:


$$\min_\theta \mathbb{E}_{p(t)p_d(x_0)q(x_t|x_0)} \left[ w(t) \lVert x_\theta(x_t,t) - x_0 \rVert^2 \right]$$


The network implicitly learns to approximate the score function as $\nabla_{x_t}\log q(x_t) \approx \frac{\alpha_t x_\theta(x_t, t) - x_t}{\sigma_t^2}$, effectively solving the ODE and completing the generation process.


## 5

Mathematical Modifications for BridgeDrive:

From Standard Diffusion to Diffusion Bridge

The Symmetrical Objective Function

To build the "Diffusion Bridge" and fix the asymmetry issue of previous methods, the model applies three critical mathematical modifications to the standard diffusion framework:

1. Replacing the Pure Noise Endpoint

2. Rewriting the Forward SDE

3. Deriving the Analytical Gaussian Transition Kernel

4. The Symmetrical Objective Function

Instead of terminating the forward process at pure Gaussian noise, BridgeDrive anchors the endpoint to the coarse expert trajectory $y$:

  $$x_T := y$$

To guarantee that the state precisely reaches the anchor $y$ at time $T$, a drift term (based on Doob's h-transform) is injected into the standard SDE, formulating the conditional Diffusion Bridge SDE:

  $$dx_t = \left[ f(t)x_t + g(t)^2 \nabla_{x_t} \log q(x_T|x_t) \right] dt + g(t)dw_t, \quad x_0 \sim p_d, x_T = y$$

Since the modified SDE remains linear, the intermediate state $x_t$ maintains a closed-form Gaussian distribution. It effectively acts as a direct interpolation between the ground truth $x_0$ and the anchor $x_T$:

  $$q(x_t|x_0,x_T) = \mathcal{N}(x_t | a_t x_T + b_t x_0, c_t^2 I)$$

With the bridge established, the neural network denoiser $x_\theta$ learns to predict the pristine trajectory $x_0$, conditioned on both the anchor $x_T$ and the environment context $z$. This formulation perfectly aligns the forward and reverse processes:

  $$\min_\theta \mathbb{E} \left[ w(t) \lVert x_\theta(x_t, t, x_T, z) - x_0 \rVert^2 \right]$$

During the inference (planning) stage, the model translates the chosen anchor $x_T$ into a refined trajectory $x_0$ using a specialized Bridge Probability Flow ODE (PF-ODE):

  $$\frac{dx_t}{dt}=f(t)x_t-g(t)^2 \left( \frac{\nabla_{x_t}\log q(x_t|x_T,z)}{2}-\nabla_{x_t}\log q(x_T|x_t) \right)$$

The trained denoiser $x_\theta(x_t, t, x_T, z)$ is utilized to approximate the score function. This score guides a numerical ODE solver (e.g., DDIM) to iteratively update the trajectory, efficiently bridging the coarse anchor to the precise final plan. 

## 6

End-to-end autonomous driving: map raw sensory inputs directly to trajectory predictions or control commands.

Deterministic planners: fusing multi modal sensor inputs through transformer-based encoders and decoding them into trajectory outputs via compact MLP heads. These models highlight the importance of effective sensor fusion in improving closed-loop driving performance.

Diffusion-based planners: utilizing to model the multi-modal nature of human driving behaviors, generating diverse and feasible trajectories from random noise through an iterative denoising process.

Patterns collapse: When patterns collapse, different random noise inputs tend to converge to similar trajectories during the denoising process.

————————————————

Out-of distribution failure scenario: despite BridgeDrive's exceptional modeling capabilities, it cannot generalize to out-of-distribution (OOD) scenarios.
