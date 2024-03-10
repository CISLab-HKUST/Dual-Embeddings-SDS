# Stable Score Distillation Via Dual Embedding
:fire::fire:**A novel method for identity driven 3D Avatar Generation**:fire::fire:
This repo is based on [Cross Initialization](https://github.com/lyuPang/CrossInitialization) and [DreamFusion](https://dreamfusion3d.github.io/), thanks for their works.
(In the future, we will rely on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting))

## Update
- [x] Release ID Embeddings Training Code (Based on [Cross Initialization](https://github.com/lyuPang/CrossInitialization))
- [x] Release Dual Embeddings Score Distillation Sampling Loss (DE-SDS)
- [ ] Release Gaussian Splatting Based 3D Avatar Optimization
- [ ] Release the Whole Identity Driven Text to 3D Avatar Generation Framework
- [ ] Release Paper and Project

## Setup
Our code **(Until Now)** mainly based on [diffusers](https://github.com/huggingface/diffusers) library.
To set up the environment, please run:
```
conda create -n de-sds python=3.9
conda activate de-sds

pip install -r requirements.txt
```

## Pipeline
### 1. Get the identity embeddings
If you have already pretrained identity text embeddings, set the `id_embeds_path` of `GuidanceParams` in the config file `configs/sample.yaml` to your pretrained identity path.

Else, put the identity image to a directory eg.`id_images/28017`, and then set the `train_data_dir` of `TrainEmbeddingParams` in the config file `configs/sample.yaml` to your this *directory*.

### 2. Models
Your may download the `Stable-Diffusion-2-1-Base` model from [link](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and put it to the `pretrained_model_name_or_path` of `TrainEmbeddingParams` and `base_model_path` of `GuidanceParams` in the config file as mentioned above.

You can also download the `ControlNet-OpenPose` model from [link](https://huggingface.co/thibaud/controlnet-sd21-openposev2-diffusers) for the future use.

### 3. 2D playground
The 3D Avatar training code is not prepared, but a 2D playground has been released.
See `2d_playground.py` for more details.


### 4. Explanations about Dual Embeddings Score Distillation Sampling (DE-SDS)
#### SDS Loss
The original form of SDS Loss as proposed in DreamFusion is:
$$
\begin{equation}
    \nabla_{\theta} \mathcal{L}_{\mathrm{SDS}}(g(\theta))=\mathbb{E}_{t,\epsilon, \mathbf{c}}\left[w(t) \frac{\sigma_{t}}{\alpha_{t}} \nabla_{\theta} \operatorname{KL}\left(q\left(\mathbf{x}_{t}|\mathbf{x}=g(\theta; \mathbf{c})\right) || \ p_{\phi}\left(\mathbf{x}_{t}|y\right)\right)\right]
\end{equation}
$$
In formal terms, the SDS Loss is actually a comparison of the 3D rendered image $\mathbf{x}=g(\theta; \mathbf{c})$ (where $\theta$ is the parameter of the 3D model and $\mathbf{c}$ is the camera pose) with the image generated based on the text $y$ in Stable Diffusion, denoted as $p_{\phi}\left(\mathbf{x}_{t}|y\right)$, in terms of similarity. Furthermore, the SDS Loss can be written in the following commonly used form:
$$
\begin{equation}
    \nabla_{\theta} \mathcal{L}_{\mathrm{SDS}}=\mathbb{E}_{t,\epsilon, \mathbf{c}}\left[w(t)((\epsilon_{\phi}(x_{t};y,t)-\epsilon)\frac{\partial{x}}{\partial{\theta}}\right]
\end{equation}
$$
This objective function can be interpreted as predicting noise input in the noisy image, and since the noise prediction of the Stable Diffusion model is based on natural images, SDS serves to guide the rendered images towards natural images. Additionally, this objective function is also related to the score matching model of [NCSN](https://arxiv.org/abs/1907.05600). NCSN is mainly designed to address the issue of low-dimensional manifolds of data distribution being concentrated in regions with a large number of data items, by introducing noise perturbations to the data distribution. The noise prediction term can actually be linked to the score function of perturbing the data distribution $q(x_{t})$:
$$
\begin{equation}
    \nabla_{x_{t}}\log q(x_{t}) \approx -\epsilon_{\phi}(x_{t};t) / \sigma_{t}
\end{equation}
$$


Currently, a implicit classifier (CFG) is often used to control the Diffusion Model to generate images that match the text description. In CFG, the predicted noise is given by:
$$
\delta_{x}(x_{t};y,t)=\underbrace{[\epsilon_{\phi}(x_{t};\empty,t) - \epsilon]}_{\delta_{x}^{gen}} + w \cdot \underbrace{[\epsilon_{\phi}(x_{t};y,t)-\epsilon_{\phi}(x_{t};\empty,t)]}_{\delta_{x}^{cls}}
$$
where $\empty$ represents empty text, and $w$ is the guidance scale used to control the degree to which the generated images match the text. Generally, as $w$ increases, the diversity of the generated images decreases, making them more consistent with the text description. SDS Loss also follows this rule.


Combining the score function interpretation of SDS with CFG, the SDS Loss can be divided into two parts, represented as $\delta_{x}:=\delta_{x}^{gen} + w \cdot \delta_{x}^{cls}$. From equation (2), it can be observed that $\delta_{x}^{gen}$ is related to $\nabla_{x_{t}}\log q(x_{t}|y)$, while $\delta_{x}^{cls}$, after derivation, is known to be related to $\nabla_{x_{t}}\log q(y|x_{t})$, which guides the data distribution of the noisy image towards a distribution that matches the text description. Therefore, this term has favorable properties, and most subsequent work only focuses on optimizing the $\delta_{x}^{gen}$ term.

#### DE-SDS
The mainly problem of $\delta_{x}^{gen}$ is mode seeking. Since $\delta_{x}^{gen}$ represent a sophisticated distribution of natural image, iteratively maximizing the likelihood between rendered image of it may leading to mode seeking promblem - finally optimized to the mean value of real distribution (More details please see [ProlificDreamer](https://arxiv.org/abs/2305.16213)). 

In DE-SDS, the $\delta_{x}^{gen}$ will be turned to:

$$
\begin{equation}
    \delta_{x}^{gen} = \epsilon_{\phi}(x_{t}; [y_{id}, y_{text}],t) - \epsilon_{\phi}(x_{t}; y_{con},t)
\end{equation}
$$

where $[y_{id}, y_{text}]$ is the pretrained id embeddings with text prompt embeddings, and $y_{con}$ is the learnable consistent embeddings optimized by the rendered image, so the term is related to the score function:

$$
\begin{equation}
    \delta_{x}^{gen} \approx \nabla_{x_{t}}\log (y_{id}|x_{t}) + \nabla_{x_{t}}\log (y_{text}|x_{t}) - \nabla_{x_{t}}\log (y_{con}|x_{t})
\end{equation}
$$

More details will be represent in the future paper.

