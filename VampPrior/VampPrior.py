from model_layers import ConvEncoder, ConvDecoder, MLP, DummyImageVec
from utils import log_Normal_diag, get_normal_nll
import torch
import torch.nn as nn
import math


class VampPrior(nn.Module):
    def __init__(self,
                 num_params,
                 in_shape,
                 n_latent_coarse,
                 n_latent_fine,
                 device,
                 hidden_dim_img_vec,
                 hidden_dim_mlp,
                 beta):
        super().__init__()
        self.device = device
        self.beta = beta
        self.n_latent_coarse = n_latent_coarse
        self.n_latent_fine = n_latent_fine
        self.num_params = num_params

        self.coarse_encoder = ConvEncoder(in_shape, n_latent_coarse)
        self.fine_encoder = DummyImageVec(in_shape, n_latent_coarse, n_latent_fine, hidden_dim_img_vec)
        self.prior_params = nn.Parameter((torch.empty(num_params, *in_shape)).normal_(0, 0.5), requires_grad=True)
        self.coarse_decoder = MLP(n_latent_coarse, n_latent_fine, hidden_dim_mlp)
        self.fine_decoder = ConvDecoder(n_latent_coarse + n_latent_fine, in_shape)
    
    def _reparametrize(self, mu_z, log_std_z, z = None):
        if z is None:
            z = torch.randn_like(input = mu_z, device = self.device)
        return mu_z + z * torch.exp(log_std_z)

    def prior(self, n: int):
        ind = torch.randint(low = 0, high = self.num_params, size = (n,))
        prior_params = self.prior_params[ind]
        mu_z, log_std_z = self.coarse_encoder(prior_params)
        z_2 = self._reparametrize(mu_z, log_std_z)
        mu_z_fine_enc, log_std_z_fine_enc = self.coarse_decoder(z_2)
        z_1 = self._reparametrize(mu_z_fine_enc, log_std_z_fine_enc)
        return torch.cat((z_1, z_2), dim=-1)
    
    def log_prior(self, z_2):
        mu_z, log_std_z = self.coarse_encoder(self.prior_params)
        mu_z, log_std_z, z_2 = mu_z.unsqueeze(0), log_std_z.unsqueeze(0), z_2.unsqueeze(1)
        a = log_Normal_diag(z_2, mu_z, log_std_z, dim=2) - math.log(self.num_params)
        a_max, _ = torch.max(a, 1)
        log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))
        return log_prior
    
    def encoder(self, x):
        mu_z_coarse, log_std_z_coarse = self.coarse_encoder(x)
        z_2 = self._reparametrize(mu_z_coarse, log_std_z_coarse)
        mu_z_fine_enc, log_std_z_fine_enc = self.fine_encoder(x, z_2)
        z_1 = self._reparametrize(mu_z_fine_enc, log_std_z_fine_enc)
        z = torch.cat((z_1, z_2), dim=-1)
        return z, mu_z_coarse, log_std_z_coarse, mu_z_fine_enc, log_std_z_fine_enc
    
    def decoder(self, z):
        mu_z_fine_dec, log_std_z_fine_dec = self.coarse_decoder(z[:,-self.n_latent_coarse:])
        mu_x = self.fine_decoder(z)
        return mu_x, mu_z_fine_dec, log_std_z_fine_dec

    def forward(self, x):
        z, mu_z_coarse, log_std_z_coarse, mu_z_fine_enc, log_std_z_fine_enc = self.encoder(x)
        mu_x, mu_z_fine_dec, log_std_z_fine_dec = self.decoder(z)
        return mu_x, z, mu_z_coarse, log_std_z_coarse, mu_z_fine_enc, log_std_z_fine_enc, mu_z_fine_dec, log_std_z_fine_dec

    def loss(self, x):
        mu_x, z, mu_z_coarse, log_std_z_coarse, mu_z_fine_enc, log_std_z_fine_enc, mu_z_fine_dec, log_std_z_fine_dec = self(x)
        z_1, z_2 = torch.chunk(z, chunks=2, dim=-1)
        log_std_x = torch.zeros((x.size(0), 1, 1, 1), device = self.device)
        nll = get_normal_nll(x, mu_x, log_std_x).flatten(1).sum(1)

        log_p_z1 = log_Normal_diag(z_1, mu_z_fine_dec, log_std_z_fine_dec, dim = 1)
        log_q_z1 = log_Normal_diag(z_1, mu_z_fine_enc, log_std_z_fine_enc, dim = 1)
        log_p_z2 = self.log_prior(z_2)
        log_q_z2 = log_Normal_diag(z_2, mu_z_coarse, log_std_z_coarse, dim = 1)
        kl = - (log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
        elbo_loss = nll + self.beta * kl
        return {'elbo_loss': elbo_loss.mean()}

    def _denormalize(self, x):
        return torch.clamp(x, -1, 1) * 0.5 + 0.5

    def sample(self, n):
        with torch.no_grad():
            z = self.prior(n)
            x = self.fine_decoder(z)
        return self._denormalize(x).cpu().numpy()
    