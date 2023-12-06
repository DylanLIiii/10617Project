# Application of Autoencoders in Fluid Dynamics for Modal Decomposition

In this study, we evaluated Multi-Layer Perceptron Autoencoders (MLP-AEs), Convolutional Neural Network Autoencoders (CNN-AEs), and Variational Autoencoders (VAEs) in the application of fluid dynamics, focusing on reconstruction accuracy, orthogonality, and interpretability.
Our findings indicate that a latent dimension of 5 optimizes performance across architectures, with CNN-AEs showing particular prowess in processing local features.
The integration of Deep Matrix Factorization (DMF) with CNN-AEs notably enhanced interpretability, especially in complex fluid dynamics, though this diminishes in subsequent modes.
VAEs, while approximating orthogonality, faced challenges in accurate representation due to data characteristics and stability issues like vanishing Kullback-Leibler divergence.
This research highlights the potential of CNN-AEs with DMF in fluid dynamics, while underscoring the limitations of VAEs in this context.

## This is offical repo of the study

## How to use it

1. Install dependencies: `conda creat -f conda-environment.yaml`
2. Run training script in created env: `python main.py --model MLP \ --lr 0.0001 \ --epochs 100 \ --batch_size 25 \ --latent_dim 5 \ --ranger21 True \ --lr_patience 10 \ --stop_patience 20 \ --deep_matrix True \ --deepth 8 \ --vae False \ --cycle 10 \ --vae_bn 0 \ --name MLPDM_latent5 \ --wandb False \ --notes test \ --tags None \ --model_dir [Your Path] `
3. This is a example command line. Using `python main.py -help` for more information.
