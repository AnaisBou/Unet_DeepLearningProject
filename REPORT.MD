# Understanding and Implementing Diffusion Models for Image Generation on CelebA

## Introduction
This report explores the implementation of **diffusion models** for image generation using the CelebA dataset. Diffusion models are a class of probabilistic generative models that progressively add noise to an image and then learn to reverse this process to generate new samples. The focus of this work is on understanding how the stochastic nature of the forward and reverse processes can be leveraged to generate high-quality images, with a special emphasis on using **U-Net architectures** for feature extraction and reconstruction.

---

## Theoretical Background

### Diffusion Models
Diffusion models are generative frameworks based on the idea of iteratively adding noise to data (forward process) and then reversing this process to generate new samples. The model learns to undo the noise added in the forward diffusion process through a series of denoising steps in the reverse process.

#### Forward Diffusion Process
The forward process progressively corrupts an input image \( x_0 \) by adding Gaussian noise at each time step \( t \). At time step \( t \), the image \( x_t \) is the noisy version of the original image. The forward process is defined as:

$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon_t,
$$

where:
- \( x_t \) is the noisy image at time step \( t \),
- \( \alpha_t \) controls the scale of the noise at time step \( t \),
- \( \epsilon_t \sim \mathcal{N}(0, I) \) is Gaussian noise.

#### Reverse Diffusion Process
The reverse diffusion process is responsible for denoising the image. It uses the learned score function \( \nabla_x \log p_t(x) \), which approximates the gradient of the log probability of the image at time step \( t \). The reverse process is modeled by the reverse-time SDE:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t) d\bar{W},
$$

where:
- \( \nabla_x \log p_t(x) \) is the score function, representing the direction in which the image should move to reduce noise,
- \( d\bar{W} \) is the reverse Wiener process.

The score function \( \nabla_x \log p_t(x) \) is learned by training a neural network to predict the noise added at each step of the forward diffusion process.

### U-Net Architecture
The U-Net architecture is widely used for tasks involving image-to-image translation, such as denoising, segmentation, and image generation. It consists of:
1. **Contracting Path**: This part of the network extracts hierarchical features using convolutions and downsampling operations.
2. **Expansive Path**: This part reconstructs the image using upsampling and skip connections, which combine information from the contracting path.

Mathematically, the output feature map at layer \( l \) in the expansive path is computed as:

$$
h_l = \text{ReLU}(\text{Conv}(h_{l-1}) + \text{Skip}(h_{L-l})),
$$

where:
- \( \text{Conv} \) represents convolutional layers,
- \( \text{Skip}(h_{L-l}) \) is the feature map from the corresponding contracting layer, which provides context and fine-grained details during reconstruction.

---

## Dataset Preparation
The **CelebA dataset** consists of over 200,000 aligned facial images with 40 attribute labels, making it ideal for training generative models that can create realistic human faces. Preprocessing steps include:
1. **Resizing**: Images are resized to \( 64 \times 64 \) pixels to reduce the computational load during training.
2. **Normalization**: Pixel values are scaled to the range \([-1, 1]\) by subtracting the mean and dividing by the standard deviation of each channel.
3. **Augmentation**: Random horizontal flips are applied to increase the diversity of the dataset and improve the model's generalization.

---

## Model Implementation

### Time Embedding (No Gaussian Fourier Projection)
To encode the time step \( t \) into a high-dimensional representation, we directly use sinusoidal embeddings (common in diffusion models). The time \( t \) is represented as:

$$
\phi(t) = [\sin(W \cdot t), \cos(W \cdot t)],
$$

where \( W \) is a matrix of fixed weights.

These time embeddings allow the model to capture temporal dependencies in the diffusion process.

### U-Net Based Score Network
The score function \( s_\theta(x_t, t) \) is parameterized as a neural network based on the U-Net architecture. The network takes the noisy image \( x_t \) and the time embedding \( \phi(t) \) as input and predicts the noise added to the image. The network output is the predicted score \( s_\theta(x_t, t) \), which guides the reverse diffusion process.

The forward pass through the score network is given by:

$$
s_\theta(x_t, t) = \text{U-Net}(x_t, \phi(t)),
$$

where:
- \( x_t \) is the noisy input image,
- \( \phi(t) \) represents the time-dependent embedding.

### Reverse-Time SDE Solver
The reverse-time SDE solver is used to generate new samples from the noise by iteratively applying the reverse diffusion process. Using the **Euler-Maruyama method**, the update step for the image at time step \( t - \Delta t \) is:

$$
x_{t-\Delta t} = x_t + \left[f(x_t, t) - g(t)^2 s_\theta(x_t, t)\right] \Delta t + g(t) \sqrt{\Delta t} z,
$$

where:
- \( z \sim \mathcal{N}(0, I) \) is random noise drawn from a standard normal distribution,
- \( g(t) \) is the diffusion coefficient,
- \( s_\theta(x_t, t) \) is the predicted score.

---

## Training Procedure

### Loss Function
The training objective minimizes the difference between the predicted noise and the true noise added during the forward diffusion process:

$$
\mathcal{L} = \mathbb{E}_{x, t, z} \left[\left\| s_\theta(x_t, t) + \frac{z}{\sigma_t} \right\|_2^2\right],
$$

where:
- \( x_t = x_0 + z \cdot \sigma_t \) is the perturbed image,
- \( z \sim \mathcal{N}(0, I) \) is Gaussian noise,
- \( \sigma_t \) is the noise scale at time \( t \).

### Training Workflow
The training process involves the following steps:
1. **Data Perturbation**: Images are perturbed by adding noise at random time steps.
2. **Score Prediction**: The model predicts the noise added during the forward diffusion process.
3. **Optimization**: The model's parameters are updated to minimize the loss function using an optimizer such as Adam.

---

## Results and Analysis

### Generated Images
The trained model generates high-quality and diverse facial images that exhibit realistic features and attribute variations. Generated samples include diverse poses, expressions, and lighting conditions.

### Evaluation Metrics
The model's performance is evaluated using:
- **Fréchet Inception Distance (FID)**: Measures the distance between the distribution of generated images and real images in feature space. A lower FID indicates better quality.
- **Diversity Score**: Quantifies the variation among generated images. A higher score indicates more diversity in the generated samples.

| Metric       | Value       |
|--------------|-------------|
| FID (↓)      | **XX.XX**   |
| Diversity (↑)| **XX.XX**   |

---

## Conclusion and Future Work

### Conclusion
This project demonstrates the effectiveness of diffusion models for generating high-quality images. The integration of the U-Net architecture and time embeddings enables the model to generate diverse and realistic facial images.

### Future Work
1. **Higher Resolution Generation**: Extend the framework to generate high-resolution images (e.g., \( 128 \times 128 \)).
2. **Faster Sampling**: Explore efficient sampling techniques to speed up the reverse process, such as using adaptive time step selection.
3. **Cross-Domain Applications**: Apply the model to other domains, such as artwork or landscape generation, to evaluate its generalization ability.

---

## References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
2. Song, Y., & Ermon, S. (2020). Score-Based Generative Modeling through SDEs. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).
3. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

---
