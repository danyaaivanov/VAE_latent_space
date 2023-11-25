from functools import reduce
import torch
import torch.nn as nn

n_linear = lambda size: reduce(lambda x, y: x * y, size)
NORM_LAYER = nn.InstanceNorm2d

class ConvEncoder(nn.Module):
    def __init__(self, input_shape, n_latent):
        super().__init__()
        assert len(input_shape) == 3
        space_shape = input_shape[1:]
        self.n_latent = n_latent

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size = 3, padding = 1),
            NORM_LAYER(32),
            nn.PReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1),
            NORM_LAYER(32),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            NORM_LAYER(64),
            nn.PReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            NORM_LAYER(64),
            nn.PReLU()
        )
        space_shape = self._shape_after_conv(space_shape=space_shape, stride=2, pad=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            NORM_LAYER(64),
            nn.PReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            NORM_LAYER(64),
            nn.PReLU()
        )
        space_shape = self._shape_after_conv(space_shape=space_shape, stride=2, pad=1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            NORM_LAYER(128),
            nn.PReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, stride = 2),
            NORM_LAYER(128),
            nn.PReLU()
        )
        space_shape = self._shape_after_conv(space_shape=space_shape, stride=2, pad=1)

        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * n_linear(space_shape), 2 * n_latent)
            )

    def forward(self, x):
        for oper in self.convs:
          x = oper(x)
        x = self.fc(x)
        mu_z, log_std_z = x[:, :self.n_latent], x[:, self.n_latent:]
        return mu_z, log_std_z

    @staticmethod
    def _shape_after_conv(space_shape, pad, stride, kernel_size=3):
       f = lambda x: int((x - kernel_size + 2 * pad) / stride) + 1
       return map(f, space_shape)

class Upscale(nn.Module):
  def __init__(self, n_latent, out_features):
    super().__init__()
    self.conv = nn.Sequential(
            nn.Conv2d(in_channels = n_latent, out_channels = 4 * out_features, kernel_size = 3, padding = 1),
            NORM_LAYER(4 * out_features),
            nn.PReLU()
        )
  def forward(self, x):
    x = self.conv(x)
    b, c, h, w = x.size()
    x = x.reshape(b, c // 4, 2, 2, h, w)
    x = torch.moveaxis(x, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    x = x.reshape(b, c // 4, h * 2, w * 2)
    return x

class ConvDecoder(nn.Module):
    def __init__(self, n_latent, output_shape):
        super().__init__()
        self.n_latent = n_latent
        self.output_shape = output_shape

        self.base_size = (128, output_shape[1] // 8, output_shape[2] // 8)

        self.fc = nn.Sequential(
            nn.Linear(n_latent, n_linear(self.base_size)),
            nn.PReLU()
            )

        self.conv1 = nn.Sequential(
            Upscale(128, 128),
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1),
            NORM_LAYER(64),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            Upscale(64, 64),
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1),
            NORM_LAYER(32),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            Upscale(32, 32),
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1),
            NORM_LAYER(16),
            nn.PReLU()
        )

        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = output_shape[0], kernel_size = 3, padding = 1)

        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, z):
        z = self.fc(z).reshape(-1, *self.base_size)
        for oper in self.convs:
          z = oper(z)
        return z
    

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim = 20):
        super().__init__()
        self.net = nn.Sequential(
           nn.Linear(input_shape, hidden_dim),
           nn.PReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.PReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.PReLU(),
           nn.Linear(hidden_dim, 2 * output_shape)
        )

    def forward(self, x):
        return self.net(x).chunk(2, dim=-1)       


class DummyImageVec(MLP):
    def __init__(self, img_shape, n_latent, output_shape, hidden_dim = 100):
        assert len(img_shape) == 3
        input_shape = n_linear(img_shape) + n_latent
        super().__init__(input_shape, output_shape, hidden_dim)
    def forward(self, x, z):
        x = torch.cat((x.flatten(1), z), dim=-1)
        return super().forward(x)