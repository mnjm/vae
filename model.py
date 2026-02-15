import torch
from torch import nn
import torch.nn.functional as F
from math import floor
from dataclasses import dataclass, field

@dataclass
class VAEConfig:
    name: str = "vae-cnn-mnist"
    input_size: list[int] = field(default_factory=lambda: [1, 28, 28])
    latent_dim: int = 20
    conv_chls: list[int] = field(default_factory=lambda: [32, 64])
    conv_kernel_sizes: list[int] = field(default_factory=lambda: [4, 4])
    conv_strides: list[int] = field(default_factory=lambda: [2, 2])
    conv_paddings: list[int] = field(default_factory=lambda: [1, 1])
    act_fn: str = 'relu'
    kl_weight: float = 5e-4
    recon_loss_type: str = 'l2'

act_fn_map = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
}

class VAE(nn.Module):

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        assert len(config.input_size) == 3
        assert len(config.conv_chls) == len(config.conv_kernel_sizes) == len(config.conv_strides) == len(config.conv_paddings)
        assert config.act_fn in act_fn_map
        act_fn = act_fn_map[config.act_fn]

        encoder_lyrs = []
        prev_chls, enc_H, enc_W = config.input_size
        for N, k, s, p in zip(config.conv_chls, config.conv_kernel_sizes, config.conv_strides, config.conv_paddings):
            encoder_lyrs.extend([
                nn.Conv2d(prev_chls, N, kernel_size=k, padding=p, stride=s),
                nn.BatchNorm2d(N),
                act_fn()
            ])
            enc_H = floor((enc_H + 2*p - k) / s) + 1
            enc_W = floor((enc_W + 2*p - k) / s) + 1
            prev_chls = N
        self.encoder = nn.Sequential(*encoder_lyrs)

        self.enc_out_shape = (prev_chls, enc_H, enc_W)
        enc_out_dim = prev_chls * enc_H * enc_W

        self.latent_mu = nn.Linear(enc_out_dim, config.latent_dim)
        self.latent_logvar = nn.Linear(enc_out_dim, config.latent_dim)

        self.fc_decode = nn.Linear(config.latent_dim, enc_out_dim)
        decoder_lyrs = []
        rev_chls = config.conv_chls[::-1]
        rev_k = config.conv_kernel_sizes[::-1]
        rev_s = config.conv_strides[::-1]
        rev_p = config.conv_paddings[::-1]
        prev_chls = rev_chls[0]
        for i in range(len(rev_chls) - 1):
            decoder_lyrs.extend([
                nn.ConvTranspose2d(prev_chls, rev_chls[i + 1], kernel_size=rev_k[i], stride=rev_s[i], padding=rev_p[i]),
                nn.BatchNorm2d(rev_chls[i + 1]),
                act_fn()
            ])
            prev_chls = rev_chls[i + 1]

        # Final layer back to input channels
        decoder_lyrs.append(nn.ConvTranspose2d(prev_chls, config.input_size[0], kernel_size=rev_k[-1], stride=rev_s[-1], padding=rev_p[-1]))
        if config.recon_loss_type != 'bce':
            decoder_lyrs.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_lyrs)

    def forward(self, x):
        B = x.shape[0]

        enc_out = self.encoder(x)
        enc_out = enc_out.view(B, -1) # B, enc_out_dim

        mu = self.latent_mu(enc_out)
        logvar = self.latent_logvar(enc_out)

        # reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        x_prime = self.fc_decode(z)
        x_prime = x_prime.view(B, *self.enc_out_shape)
        x_prime = self.decoder(x_prime)

        loss = self.loss(x_prime, x, mu, logvar)

        return x_prime, loss

    def loss(self, x_prime: torch.tensor, x: torch.tensor, mu: torch.tensor, logvar: torch.tensor):
        recon_loss_fn = {
            'l1': F.l1_loss,
            'l2': F.mse_loss,
            'bce': F.binary_cross_entropy_with_logits,
        }[self.config.recon_loss_type]
        kl_weight = self.config.kl_weight
        recon = recon_loss_fn(x_prime, x, reduction='sum') / x_prime.size(0)
        kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()
        loss = recon + kl_weight * kl
        return {
            'loss': loss,
            'recon_loss': recon,
            'kl_loss': kl
        }

    def configure_optimizer(self, optim_cfg: object, device: torch.device):
        supported_optimizers_map = { 'adamw': torch.optim.AdamW }
        assert optim_cfg.type in supported_optimizers_map, f"f{optim_cfg.name=} optimizer not supported"
        optim_init = supported_optimizers_map[optim_cfg.type]

        optim_cfg.fused = getattr(optim_cfg, 'fused', False) and device.type == "cuda"
        weight_decay = getattr(optim_cfg, 'weight_decay', 1e-2)
        lr = optim_cfg.lr
        params_dict = { pn: p for pn, p in self.named_parameters() }
        params_dict = { pn:p for pn, p in params_dict.items() if p.requires_grad } # filter params that requires grad
        # create optim groups of any params that is 2D or more. This group will be weight decayed ie weight tensors in Linear and embeddings
        decay_params = [ p for p in params_dict.values() if p.dim() >= 2]
        # create optim groups of any params that is 1D. All biases and layernorm params
        no_decay_params = [ p for p in params_dict.values() if p.dim() < 2]
        optim_groups = [
            { 'params': decay_params, 'weight_decay': weight_decay },
            { 'params': no_decay_params, 'weight_decay': 0.0 },
        ]
        kwargs = dict(optim_cfg)
        kwargs.pop('type')
        kwargs.pop('weight_decay')
        optimizer = optim_init(optim_groups, **kwargs)
        return optimizer
