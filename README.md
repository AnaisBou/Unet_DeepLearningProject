# Unet_DeepLearningProject: Diffusion Models for Image Generation using U-Net on CelebA Dataset

## Overview
This project, **Unet_DeepLearningProject**, explores the application of **diffusion models** for image generation, specifically using the **CelebA dataset** and a **U-Net** architecture. The goal is to generate high-quality face images from noise through the reverse diffusion process. The diffusion model, inspired by **Score-Based Generative Models**, is integrated with U-Net to learn how to reverse the added noise and generate realistic images.

### Key Features:
- Implementation of a **diffusion model** with **U-Net** for image generation.
- Training and evaluation on the **CelebA dataset**, a large collection of celebrity face images.
- Use of **mixed-precision training** for efficient training.
- Generation of images from random noise using the trained model.

### Created by: **AnaisBoughanem**

## Project Structure

The project consists of the following files:

- **README.md**: This file, providing an overview of the project.
- **REPORT.md**: A detailed report explaining the theory behind U-Net and diffusion models and their application in image generation.
- **src.ipynb**: A Jupyter notebook containing the implementation of the diffusion model, U-Net architecture, and the training loop.

## Installation

To set up the environment for this project, please follow the steps below:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AnaisBou/Unet_DeepLearningProject.git
    cd Unet_DeepLearningProject
    ```

2. **Install dependencies**:
    The project requires the following Python libraries:
    - `torch`
    - `torchvision`
    - `PIL`
    - `tqdm`
    - `matplotlib`
    - `numpy`

3. **Download the CelebA dataset**:
    The CelebA dataset can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Make sure to place the dataset in the appropriate directory and update the path in the code.

## Dataset

### CelebA Dataset

The CelebA dataset consists of 20,599 celebrity images, each annotated with 40 attribute labels. These images are used for training the diffusion model to generate new face images.

**Example images from CelebA:**

![CelebA Example](CelebA1.png)

For this project, the images were resized to 256x256 pixels for training.

## Training

The training of the diffusion model with U-Net involves the following key steps:

1. **Preprocessing the CelebA images**: Resize the images to 256x256 pixels and normalize pixel values to a range suitable for neural network training.
2. **Training the Diffusion Model**: The model is trained for 10 epochs using a **Mean Squared Error (MSE)** loss function. The optimizer used is **Adam**, and **mixed-precision training** is employed to reduce computational overhead.
3. **Saving the Model**: The trained model is saved periodically to allow for checkpointing.

To run the training, execute the notebook `src.ipynb` and follow the steps in the notebook cells.

## Results

### Generated Images

Below are images generated by the trained diffusion model after 10 epochs of training:

**Generated Images:**

![Generated Images](CAgenerated.png)

As shown, the model was unable to generate clear or recognizable faces. Instead, the generated images appear as pixelated noise without any clear structure, indicating that the model struggled to reverse the added noise effectively. This suggests that the current architecture and training setup were insufficient for achieving meaningful results.

### Challenges Encountered

1. **Difficulty in Implementing the Diffusion Model**: 
   Implementing the diffusion model integrated with U-Net was more challenging than expected. Despite following theoretical resources and tutorials, the model struggled to generate meaningful images. The results were pixelated and lacked any clear structure or resemblance to faces.

2. **Inadequate Image Quality**: 
   The generated images were highly pixelated, with no recognizable patterns resembling faces. This indicates that the model was unable to reverse the added noise effectively, pointing to the need for further tuning.

3. **Training Instability**: 
   The training process was unstable, especially in the early epochs. Adjusting hyperparameters and diffusion steps helped to some extent, but the model still failed to generate sharper, more coherent images.

4. **GPU Limitations**: 
   The training process required substantial computational power, and although the model was run on a GPU, the hardware limitations may have hindered the model’s ability to converge to more realistic outputs. Further tuning or a more powerful GPU might be necessary for better results.

## References

- **U-Net Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **Score-Based Generative Modeling Paper**: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- **Denoising Diffusion Probabilistic Models**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **CelebA Dataset**: [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
