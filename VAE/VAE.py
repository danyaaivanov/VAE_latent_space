from model_layers import ConvEncoder, ConvDecoder
import torch
from torch import nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, input_shape, n_latent, beta=1):
        super(ConvVAE, self).__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.n_latent = n_latent
        self.beta = beta

        self.encoder = ConvEncoder(self.input_shape, self.n_latent)
        self.decoder = ConvDecoder(self.n_latent, self.input_shape)

    def prior(self, n, use_cuda=True):
        z = torch.randn(n, self.n_latent)
        if use_cuda:
            z = z.cuda()
        return z

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z
        x_recon = self.decoder(z)
        return x_recon, mu_z, logvar_z

    def loss(self, x):
        x_recon, mu_z, logvar_z = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        elbo_loss = recon_loss + self.beta * kl_loss
        return {
            'elbo_loss': elbo_loss,
#             'recon_loss': recon_loss,
#             'kl_loss': kl_loss
        }

    def sample(self, n):
        with torch.no_grad():
            z = self.prior(n)
            x_recon = self.decoder(z)
            samples = torch.clamp(x_recon, -1, 1)
            return samples.cpu().numpy() * 0.5 + 0.5