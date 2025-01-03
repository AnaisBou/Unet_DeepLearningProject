# Report: Understanding and Implementing Diffusion Models for Image Generation using U-Net

## 1. Introduction
This report presents the implementation of a diffusion model for image generation, leveraging the **U-Net** architecture, applied to the **CelebA** dataset. Diffusion models are a class of generative models that have gained significant attention in recent years due to their ability to generate high-quality images. In this project, we aim to explore the mechanics of diffusion models, their interaction with U-Net architectures, and how these can be applied for image synthesis tasks. Specifically, this work involves generating images of faces from the CelebA dataset, a popular benchmark for face generation tasks.

The central aim is to investigate how the diffusion process can be used to generate high-quality images and to integrate U-Net as a denoising network in this process.

## 2. Background

### 2.1. U-Net Architecture
U-Net, originally introduced in 2015 for biomedical image segmentation, has since found applications in a variety of image generation and restoration tasks. The U-Net architecture consists of two main components:
- **Encoder**: A series of convolutional layers that gradually reduce the spatial dimensions of the input while increasing the depth of feature maps.
- **Decoder**: A symmetric upsampling network that reconstructs the image, guided by the feature maps extracted in the encoder. One of the key features of U-Net is its use of **skip connections**, which allow feature maps from the encoder to be passed directly to the corresponding decoder layers. These skip connections are crucial for retaining spatial information and improving the quality of the generated output.

For generative tasks like this project, U-Net acts as the core denoising network within the reverse diffusion process, learning how to reverse the noisy transformation applied during the forward diffusion process.

### 2.2. Diffusion Models
Diffusion models represent a class of generative models inspired by **stochastic differential equations (SDEs)** and **Markov chains**. The core idea behind diffusion models is to model the process of gradually adding noise to data over a series of steps and then training a model to reverse this process.

#### 2.2.1. The Forward Process
The forward process is essentially a sequence of noise addition steps that progressively transforms a clean image into pure noise. Mathematically, it can be described as a diffusion process governed by a set of transition probabilities. The forward process is defined as gradually applying noise to the data:

\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
\]
Where \( x_t \) represents the noisy image at step \( t \), and \( \beta_t \) controls the noise schedule.

#### 2.2.2. The Reverse Process
The reverse process aims to reverse the forward process, learning how to transform noisy data back to clean data. This is where the U-Net architecture comes in, acting as a **denoising network** that predicts the noise added at each step.

### 2.3. U-Net in Diffusion Models
In this project, the **U-Net** architecture is employed in the **reverse diffusion process**. Given an image that has been progressively corrupted by noise, the model learns to predict the clean image step by step, using a modified U-Net model. The model is trained on noisy images and learns to predict both the original image and the noise that was added to it at each step.

### 2.4. CelebA Dataset
The **CelebA dataset** consists of over 200,000 celebrity face images, each with 40 attribute labels. For this project, the CelebA dataset is used for training the diffusion model, which aims to generate new face images based on the learned distribution.

## 3. Methodology

### 3.1. Dataset Preprocessing
The CelebA dataset is preprocessed by resizing the images to a consistent size of **256x256 pixels** and normalizing the pixel values. This step ensures that the model receives uniformly-sized images and that the pixel values are in a range suitable for neural network training.

### 3.2. Model Architecture
The diffusion model is composed of two major components:
1. **UNetBlock**: A basic building block of the U-Net architecture, which consists of two convolutional layers and a dropout layer to prevent overfitting.
2. **Diffusion Step**: A simple module that adds Gaussian noise to the input images to simulate the diffusion process.

The full U-Net model performs downsampling through convolutional layers and upsampling through transpose convolution layers. After each upsampling step, the network crops the corresponding features from earlier layers (skip connections) and concatenates them for better spatial resolution.

### 3.3. Training the Diffusion Model
The model is trained to reverse the noise addition process, learning how to generate images from noisy inputs. The loss function used is **Mean Squared Error (MSE)**, comparing the model's output to the clean images. Additionally, the optimizer used is **Adam**, and **mixed-precision training** is employed to improve training efficiency.

### 3.4. Evaluation
After training, the model is evaluated by generating images from random noise and comparing them to real images from the CelebA dataset. The images are qualitatively assessed to check how well the model has learned to generate realistic faces.

## 4. Results

### 4.1. Generated Images
Below are the images generated by the trained diffusion model after 10 epochs. The images demonstrate the model's ability to generate realistic faces from random noise.

![Generated Image 1](generated_images/sample_1.png)
![Generated Image 2](generated_images/sample_2.png)

### 4.2. Challenges and Solutions
1. **Training Stability**: During the initial stages of training, the model showed signs of instability, including rapid oscillations in the loss curve. This was mitigated by using **mixed-precision training** (AMP) to stabilize gradient updates and allow for a more efficient training process.
   
2. **Quality of Generated Images**: Early generations lacked sharpness and clarity. This was improved by adjusting the noise schedule (\(\beta_t\)) and increasing the training duration.

3. **Computational Requirements**: Due to the large number of parameters in the U-Net model and the high computational cost of training diffusion models, training on a **GPU** was necessary to reduce training time.

### 4.3. Analysis of Model Performance
While the generated images are relatively realistic, further improvements could be achieved by:
- Increasing the number of diffusion steps to refine the generated image.
- Experimenting with different **learning rate schedules**.
- Using more advanced architectures such as **conditional U-Nets**.

## 5. Conclusion
The implementation of the diffusion model with a U-Net architecture has shown promising results in generating realistic faces from the CelebA dataset. The model successfully learns to reverse the noise process and generate images from noisy inputs. However, there is still room for improvement, especially regarding the sharpness and detail of generated faces. Future work may include improving the training process, experimenting with different architectures, and fine-tuning hyperparameters to enhance image quality.

## 6. References
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Score-Based Generative Modeling Paper](https://arxiv.org/abs/2011.13456)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
