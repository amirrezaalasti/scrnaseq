"""
Data preprocessing utilities for single-cell RNA-seq analysis.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def generate_fixed_perturbation_labels(labels: pd.Series) -> pd.Series:
    """Generate fixed perturbation labels by removing 'ctrl+' and '+ctrl'."""
    labels_fixed = labels.str.replace(pat="ctrl+", repl="")
    labels_fixed = labels_fixed.str.replace(pat="+ctrl", repl="")
    return labels_fixed


def preprocess_data(adata, n_genes: int = 1000):
    """
    Preprocess single-cell RNA-seq data.
    
    Args:
        adata: AnnData object
        n_genes: Number of top variance genes to select
        
    Returns:
        Preprocessed AnnData object
    """
    # Fix perturbation labels
    adata.obs["condition_fixed"] = generate_fixed_perturbation_labels(
        labels=adata.obs["condition"]
    )
    
    # Filter out double-gene perturbations
    filter_mask = ~adata.obs["condition_fixed"].str.contains(r"\+")
    indexes_to_keep = filter_mask[filter_mask].index
    adata_single = adata[indexes_to_keep].copy()
    
    # Select high-variance genes
    gene_variances = adata_single.X.toarray().var(axis=0)
    sorted_indexes = gene_variances.argsort()[::-1]
    top_gene_indexes = sorted_indexes[:n_genes]
    adata_single_top_genes = adata_single[:, top_gene_indexes].copy()
    
    return adata_single_top_genes


def create_dataloaders(X, batch_size: int = 128, train_ratio: float = 0.7, 
                       val_ratio: float = 0.15, seed: int = 42, 
                       num_workers: int = 4, pin_memory: bool = True):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        X: Input tensor
        batch_size: Batch size
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        seed: Random seed
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = TensorDataset(X, X)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset, 
        lengths=[train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


