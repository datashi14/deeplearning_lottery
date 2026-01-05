import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, timesteps=300, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.device = device
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
            
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Reverse process one step: p(x_{t-1} | x_t)
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        
        # Model predicts noise eps_theta
        # mean = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_hat_t) * eps_theta)
        
        eps_theta = model(x, t)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps_theta / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16):
        """
        Full sampling loop from pure noise to x_0
        """
        device = self.device
        img = torch.randn((batch_size, 1, image_size, image_size), device=device)
        
        # Iteratively denoise
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            
        # Clamp to [-1, 1] usually
        img = torch.clamp(img, -1.0, 1.0)
        return img
