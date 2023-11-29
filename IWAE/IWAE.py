from model_layers import ConvEncoder, ConvDecoder
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class IWAE(nn.Module):
    def __init__(self, input_shape, n_latent, K = 1, beta =1, device = 'cuda'):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.device = device
        self.n_latent = n_latent
        self.K = K
        self.beta = beta
        self.encoder = ConvEncoder(self.input_shape, self.n_latent)
        output_shape = [input_shape[0]*2, input_shape[1], input_shape[2]]
        self.decoder = ConvDecoder(self.n_latent, output_shape)

    def prior(self, n, use_cuda=True):

        z = torch.randn(n, self.n_latent).cuda()
        if use_cuda:
            z = z.cuda()
        return z

    def forward(self, x, train = False):
        mu_z, log_std_z = self.encoder(x)
        q_z = torch.distributions.Normal(loc=mu_z, scale=log_std_z.exp())
        if train:
            z = q_z.rsample(torch.Size([self.K]))  #[K, Batch, latent]
        else:
            z = q_z.rsample()
        mu_x, log_std_x = self.decoder(z.view(-1, self.n_latent)).view(self.K, *x.shape).chunk(2, dim = 1) #should std also be separate entity?
        return mu_x, mu_z, log_std_x, log_std_z, z


    def loss(self, x):
        mu_x, mu_z, log_std_x, log_std_z, z = self.forward(x, train = True)
        pz = Independent(Normal(loc=torch.zeros_like(mu_z).to(self.device),
                            scale=torch.zeros_like(log_std_z).exp().to(self.device)),
                     reinterpreted_batch_ndims=1)
        qz_x = Independent(Normal(loc=mu_z,
                              scale=torch.exp(log_std_z).to(self.device)),
                       reinterpreted_batch_ndims=1)

        kl_loss = torch.mean(torch.logsumexp(qz_x.log_prob(z) - pz.log_prob(z), 0))

        x_z = Independent(Normal(loc=mu_x,
                              scale=torch.exp(log_std_x)), #torch.zeros_like(mu_x)
                       reinterpreted_batch_ndims=0)
        
        recon_loss = -torch.mean(torch.logsumexp(x_z.log_prob(torch.tile(x, dims = [self.K, 1, 1, 1, 1])).sum([-1, -2, -3]), 0))

        return {
            'elbo_loss': recon_loss + self.beta*kl_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def sample(self, n):
        with torch.no_grad():
            x_recon = self.decoder(self.prior(n))
            samples = torch.clamp(x_recon, -1, 1)
        return samples.cpu().numpy() * 0.5 + 0.5