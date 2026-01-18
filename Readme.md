# Single-Cell RNA-Seq Analysis with Deep Learning

This project is part of the course **Applied Machine Learning in Genomic Data Science (WiSe 2025/26)** at **Leibniz University of Hannover**.

## Overview

This project implements and compares three deep learning architectures for single-cell RNA sequencing (scRNA-seq) data analysis:

1. **Autoencoder (AE)**: A standard autoencoder for dimensionality reduction and feature learning
2. **Variational Autoencoder (VAE)**: A probabilistic autoencoder that learns a latent distribution
3. **Generative Adversarial Network (GAN)**: A generative model for synthesizing single-cell expression profiles

## Project Structure

- `models.py`: Contains the PyTorch Lightning implementations of all three model architectures
- `data_utils.py`: Data preprocessing utilities for scRNA-seq data, including perturbation label handling and dataloader creation
- `evaluation.py`: Evaluation metrics and utilities for model assessment
- `scrnaseq_project.ipynb`: Main notebook for running experiments and analysis
- `data/`: Directory containing the Norman perturbation dataset

## Models

### Autoencoder
A standard encoder-decoder architecture with batch normalization and dropout for regularization. The model learns compressed representations of single-cell expression profiles.

### Variational Autoencoder
An extension of the autoencoder that learns a probabilistic latent space. Includes KL divergence annealing for stable training and better latent representations.

### Generative Adversarial Network
A GAN architecture with spectral normalization for training stability, gradient penalty, and sparsity regularization to match the zero-inflated nature of scRNA-seq data.

## Data

The project uses the Norman perturbation dataset, which contains single-cell RNA-seq data with genetic perturbations. The preprocessing pipeline:
- Filters single-gene perturbations
- Selects high-variance genes
- Splits data into train/validation/test sets

## Dependencies

- PyTorch
- PyTorch Lightning
- NumPy
- Pandas
- scikit-learn
- scipy
- AnnData (for single-cell data handling)

## Usage

The main workflow is implemented in `scrnaseq_project.ipynb`, which includes:
- Data loading and preprocessing
- Model training with PyTorch Lightning
- Model evaluation and comparison
- Visualization of results
