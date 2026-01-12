"""
Deep learning models for single-cell RNA-seq analysis.
Includes Autoencoder, VAE, and GAN architectures with smaller, efficient designs.
"""

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network for Autoencoder."""
    def __init__(self, in_features: int, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Decoder network for Autoencoder."""
    def __init__(self, latent_dim: int = 64, out_features: int = 1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_features)
        )
    
    def forward(self, x):
        return self.net(x)


class Autoencoder(pl.LightningModule):
    """Autoencoder for dimensionality reduction."""
    def __init__(self, in_features: int, latent_dim: int = 64, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(in_features, latent_dim)
        self.decoder = Decoder(latent_dim, in_features)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class VAE(pl.LightningModule):
    """Variational Autoencoder with KL annealing."""
    def __init__(self, in_features: int, latent_dim: int = 64, learning_rate: float = 1e-3, 
                 kl_weight: float = 0.1, kl_annealing: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, in_features)
        )
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def get_kl_weight(self):
        if self.kl_annealing:
            progress = min(self.current_epoch / (self.trainer.max_epochs * 0.5), 1.0)
            return self.kl_weight * progress
        return self.kl_weight
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kl_w = self.get_kl_weight()
        loss = recon_loss + kl_w * kl_loss
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("recon_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("kl_loss", kl_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("kl_weight", kl_w, prog_bar=False, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kl_w = self.get_kl_weight()
        loss = recon_loss + kl_w * kl_loss
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + self.kl_weight * kl_loss
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("test_kl_loss", kl_loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class Generator(nn.Module):
    """Generator network for GAN with increased capacity for complex distributions."""
    def __init__(self, latent_dim: int, out_features: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, out_features)
        )
        
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator network for GAN with spectral normalization for stability."""
    def __init__(self, in_features: int, use_spectral_norm: bool = True):
        super().__init__()
        from torch.nn.utils import spectral_norm
        
        if use_spectral_norm:
            self.model = nn.Sequential(
                spectral_norm(nn.Linear(in_features, 512)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                spectral_norm(nn.Linear(512, 256)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                spectral_norm(nn.Linear(256, 128)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
                spectral_norm(nn.Linear(128, 1)),
                nn.Sigmoid()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        return self.model(x)


class GAN(pl.LightningModule):
    """Generative Adversarial Network with gradient penalty and sparsity regularization for scRNA-seq."""
    def __init__(self, in_features: int, latent_dim: int = 64, learning_rate: float = 2e-4, 
                 n_discriminator_steps: int = 5, sparsity_weight: float = 0.01, 
                 label_smoothing: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim, in_features)
        self.discriminator = Discriminator(in_features, use_spectral_norm=True)
        self.automatic_optimization = False
        self.n_discriminator_steps = n_discriminator_steps
        self.sparsity_weight = sparsity_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, z):
        return self.generator(z)
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP style training."""
        alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
        
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        opt_g, opt_d = self.optimizers()
        
        z = torch.randn(x.size(0), self.hparams.latent_dim, device=x.device)
        
        # Initialize variables for logging
        d_loss = torch.tensor(0.0, device=x.device)
        gradient_penalty = torch.tensor(0.0, device=x.device)
        
        # Train discriminator multiple times
        for _ in range(self.n_discriminator_steps):
            self.toggle_optimizer(opt_d)
            
            # Add small noise to real samples for better generalization
            x_noisy = x + torch.randn_like(x) * 0.01
            real_validity = self.discriminator(x_noisy)
            
            # Label smoothing for real samples (prevents overconfidence)
            real_labels = torch.ones_like(real_validity) * (1.0 - self.label_smoothing)
            real_loss = F.binary_cross_entropy(real_validity, real_labels)
            
            fake_samples = self.generator(z).detach()
            fake_validity = self.discriminator(fake_samples)
            fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
            
            gradient_penalty = self.compute_gradient_penalty(x, fake_samples)
            
            d_loss = (real_loss + fake_loss) / 2 + 0.1 * gradient_penalty
            
            self.manual_backward(d_loss)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)
        
        # Train generator
        self.toggle_optimizer(opt_g)
        fake_samples = self.generator(z)
        fake_validity = self.discriminator(fake_samples)
        g_loss = F.binary_cross_entropy(fake_validity, torch.ones_like(fake_validity))
        
        # Add sparsity penalty to encourage zero-inflation (important for scRNA-seq)
        # This encourages the generator to produce zeros, matching real data distribution
        sparsity_penalty = self.sparsity_weight * torch.mean(torch.abs(fake_samples))
        g_loss = g_loss + sparsity_penalty
        
        self.log("g_loss", g_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("d_loss", d_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("gradient_penalty", gradient_penalty, prog_bar=False, on_step=False, on_epoch=True)
        self.log("sparsity_penalty", sparsity_penalty, prog_bar=False, on_step=False, on_epoch=True)
        
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = torch.randn(x.size(0), self.hparams.latent_dim, device=x.device)
        
        real_validity = self.discriminator(x)
        fake_samples = self.generator(z)
        fake_validity = self.discriminator(fake_samples)
        
        real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
        fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
        d_loss = (real_loss + fake_loss) / 2
        
        g_loss = F.binary_cross_entropy(fake_validity, torch.ones_like(fake_validity))
        
        self.log("val_d_loss", d_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_g_loss", g_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return d_loss
    
    def test_step(self, batch, batch_idx):
        """Test step for GAN evaluation."""
        x, _ = batch
        z = torch.randn(x.size(0), self.hparams.latent_dim, device=x.device)
        
        real_validity = self.discriminator(x)
        fake_samples = self.generator(z)
        fake_validity = self.discriminator(fake_samples)
        
        real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
        fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
        d_loss = (real_loss + fake_loss) / 2
        
        g_loss = F.binary_cross_entropy(fake_validity, torch.ones_like(fake_validity))
        
        self.log("test_d_loss", d_loss, on_step=False, on_epoch=True)
        self.log("test_g_loss", g_loss, on_step=False, on_epoch=True)
        
        return d_loss
    
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        return [opt_g, opt_d], []

