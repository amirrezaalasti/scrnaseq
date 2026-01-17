"""
Evaluation utilities for model assessment.
"""

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def evaluate_model(model, test_loader, model_type='ae', scaler=None):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        model_type: Type of model ('ae', 'vae', or 'gan')
        scaler: Optional sklearn scaler for inverse transformation (for log1p+standardized data)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_reconstructed = []
    all_original = []
    all_latent = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            if model_type == 'ae':
                latent = model.encoder(x)
                reconstructed = model(x)
            elif model_type == 'vae':
                mu, log_var = model.encode(x)
                latent = mu
                reconstructed, _, _ = model(x)
            elif model_type == 'gan':
                # For GAN, we generate samples but don't have reconstruction
                continue
            else:
                continue
            
            all_reconstructed.append(reconstructed.cpu().numpy())
            all_original.append(x.cpu().numpy())
            all_latent.append(latent.cpu().numpy())
    
    if model_type == 'gan':
        return None
    
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    original = np.concatenate(all_original, axis=0)
    latent = np.concatenate(all_latent, axis=0)
    
    # Inverse transform if scaler is provided (for log1p + StandardScaler normalization)
    if scaler is not None:
        print("  ✓ Applying inverse transformation (expm1 + inverse scaling)")
        original = np.expm1(scaler.inverse_transform(original))
        reconstructed = np.expm1(scaler.inverse_transform(reconstructed))
        # Clip negative values from numerical instability
        original = np.maximum(original, 0)
        reconstructed = np.maximum(reconstructed, 0)
    
    mse = mean_squared_error(original, reconstructed)
    mae = mean_absolute_error(original, reconstructed)
    r2 = r2_score(original, reconstructed)
    
    mse_per_gene = mean_squared_error(original, reconstructed, multioutput='raw_values')
    r2_per_gene = r2_score(original, reconstructed, multioutput='raw_values')
    
    corr_per_gene = []
    for i in range(original.shape[1]):
        corr, _ = pearsonr(original[:, i], reconstructed[:, i])
        corr_per_gene.append(corr)
    corr_per_gene = np.array(corr_per_gene)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mse_per_gene': mse_per_gene,
        'r2_per_gene': r2_per_gene,
        'corr_per_gene': corr_per_gene,
        'latent': latent,
        'reconstructed': reconstructed,
        'original': original
    }


def generate_synthetic_data(gan_model, n_samples: int, latent_dim: int = 64):
    """
    Generate synthetic data using a trained GAN.
    
    Args:
        gan_model: Trained GAN model
        n_samples: Number of samples to generate
        latent_dim: Latent dimension
        
    Returns:
        Generated synthetic data as numpy array
    """
    gan_model.eval()
    with torch.no_grad():
        device = next(gan_model.parameters()).device
        z = torch.randn(n_samples, latent_dim, device=device)
        synthetic_data = gan_model(z).cpu().numpy()
    return synthetic_data


