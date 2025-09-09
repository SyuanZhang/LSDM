import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SignalDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.task_id = params.task_id
        self.input_dim = params.sample_rate
        self.extra_dim = params.extra_dim
        self.max_step = params.max_step

        beta = torch.tensor(np.array(params.noise_schedule, dtype=np.float32))   # [T]
        alpha = 1.0 - beta                                                       # [T]
        alpha_bar = torch.cumprod(alpha, dim=0)                                  # [T]

        var_blur = torch.tensor(np.array(params.blur_schedule, dtype=np.float32))      # [T]
        var_blur_bar = torch.cumsum(var_blur, dim=0)                                   # [T]
        var_kernel = (self.input_dim / var_blur).unsqueeze(1)                          # [T,1]
        var_kernel_bar = (self.input_dim / var_blur_bar).unsqueeze(1)                  # [T,1]

        # 注册为 buffer —— 这些会随 module.to(device) 一起移动
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('var_blur', var_blur)
        self.register_buffer('var_blur_bar', var_blur_bar)
        self.register_buffer('var_kernel', var_kernel)
        self.register_buffer('var_kernel_bar', var_kernel_bar)

        gk = self.get_kernel(self.var_kernel)                 # [T,N]
        gk_bar = self.get_kernel(self.var_kernel_bar)         # [T,N]
        self.register_buffer('gaussian_kernel', gk)
        self.register_buffer('gaussian_kernel_bar', gk_bar)

        info_w = gk_bar * torch.sqrt(self.alpha_bar).unsqueeze(-1)    # [T,N]
        self.register_buffer('info_weights', info_w)

        nw = self.get_noise_weights()                          # [T,N]
        self.register_buffer('noise_weights', nw)

    def get_kernel(self, var_kernel):
        # 确保 samples 建在与 var_kernel 同一设备
        samples = torch.arange(self.input_dim, device=var_kernel.device, dtype=var_kernel.dtype)  # [N]
        # 形状对齐: var_kernel [T,1]，需要广播到 [T,N]
        x = (samples - self.input_dim // 2) ** 2                               # [N]
        denom = 2.0 * var_kernel                                               # [T,1]
        num = torch.exp(-x.unsqueeze(0) / denom)                               # [T,N]
        norm = torch.sqrt(2.0 * torch.pi * var_kernel)                         # [T,1]
        gaussian_kernel = num / norm                                           # [T,N]
        gaussian_kernel = self.input_dim * gaussian_kernel / gaussian_kernel.sum(dim=1, keepdim=True)
        return gaussian_kernel
    
    def get_noise_weights(self):
        ws = []
        T = self.max_step
        for t in range(T):
            upper = t + 1
            one_minus_alpha_sqrt = torch.sqrt(1.0 - self.alpha[:upper])                 # [t]
            rev_one_minus_alpha_sqrt = torch.flip(one_minus_alpha_sqrt, dims=[0])       # [t]
            rev_alpha = torch.flip(self.alpha[:upper], dims=[0])                        # [t]
            rev_alpha_bar_sqrt = torch.sqrt(torch.cumprod(rev_alpha, dim=0) / rev_alpha[-1])  # [t]

            rev_var_blur = torch.flip(self.var_blur[:upper], dims=[0])                  # [t]
            rev_var_blur_bar = torch.cumsum(rev_var_blur, dim=0) - rev_var_blur[-1]     # [t]
            rev_var_kernel_bar = (self.input_dim / rev_var_blur_bar).unsqueeze(1)       # [t,1]
            rev_kernel_bar = self.get_kernel(rev_var_kernel_bar)                         # [t,N]
            # 避免 1/0 导致的 inf：首行置 1
            rev_kernel_bar[0, :] = torch.ones(self.input_dim, device=rev_kernel_bar.device, dtype=rev_kernel_bar.dtype)

            # (rev_alpha_bar_sqrt[:,None] * rev_kernel_bar): [t,N]; 转置后 [N,t]
            # mv: [N,t] @ [t] -> [N]
            w = torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1),
                         rev_one_minus_alpha_sqrt)                                     # [N]
            ws.append(w)
        return torch.stack(ws, dim=0)  # [T,N]

    def degrade_fn(self, x_0, t, task_id):
        # 断言设备一致，调试更快定位
        assert x_0.device == self.info_weights.device == self.noise_weights.device, \
            (x_0.device, self.info_weights.device, self.noise_weights.device)
        assert t.device == self.info_weights.device, (t.device, self.info_weights.device)
        if t.dtype != torch.long:
            t = t.long()

        # [B,N] → 根据任务增维，靠广播与 x_0 相乘
        nw = self.noise_weights.index_select(0, t)    # [B,N]
        iw = self.info_weights.index_select(0, t)     # [B,N]

        # 不需要再 .to(device)，它们已经是 buffer 并在同一设备
        if task_id in (0, 1, 4):      # WiFi/FMCW/Traffic
            shape = (-1, 1, 1)
        elif task_id in (2, 3):       # MIMO/EEG
            shape = (-1, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported task_id: {task_id}")

        nw = nw.unsqueeze(-1).unsqueeze(-1).reshape(*nw.shape[:2], *([1] * (x_0.ndim - 2)))  # 自适配
        iw = iw.unsqueeze(-1).unsqueeze(-1).reshape_as(nw)

        # 注意：手动设随机种子会导致每步同噪声；通常不建议在这里固定
        noise = torch.randn_like(x_0)
        x_t = iw * x_0 + nw * noise
        return x_t

    # ... 其他方法保持不变 ...

    

class GaussianDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_dim = params.sample_rate
        self.extra_dim = params.extra_dim
        self.max_step = params.max_step

        beta = torch.tensor(np.array(params.noise_schedule, dtype=np.float32))  # [T]
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('noise_weights', torch.sqrt(1.0 - alpha_bar))      # [T]
        self.register_buffer('info_weights', torch.sqrt(alpha_bar))        

    def degrade_fn(self, x_0, t, task_id):
        assert x_0.device == self.info_weights.device == self.noise_weights.device
        if t.dtype != torch.long:
            t = t.long()

        nw = self.noise_weights.index_select(0, t)   # [B]
        iw = self.info_weights.index_select(0, t)    # [B]

        # 扩到 [B, 1, ...] 以广播乘 x_0
        # 先 [B]→[B,1]→[B,1,1]... 到与 x_0 维度一致
        while nw.ndim < x_0.ndim:
            nw = nw.unsqueeze(-1)
            iw = iw.unsqueeze(-1)

        noise = torch.randn_like(x_0)
        x_t = iw * x_0 + nw * noise
        return x_t


    def sampling(self, restore_fn, cond, device):
        """ Samples the data by reversing the diffusion process. """
        batch_size = cond.shape[0]  # B
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        
        inf_weight = (self.noise_weights[self.max_step-1] + self.info_weights[self.max_step-1]).to(device)  # scalar
        x_s = inf_weight * torch.randn(data_dim, dtype=torch.float32, device=device)  # [B, N, S, A, 2]
        
        # Restore data from noise.
        for s in range(self.max_step - 1, -1, -1):  # reverse from t to 0
            x_0_hat = restore_fn(x_s, s * torch.ones(batch_size, dtype=torch.int64), cond)  # Restore x_0 from x_s
            if s > 0:
                x_s = self.degrade_fn(x_0_hat, t=(s - 1) * torch.ones(batch_size, dtype=torch.int64))  # degrade \hat{x_0} to x_{s-1}
        
        return x_0_hat

    def robust_sampling(self, restore_fn, cond, device):
        """ Performs robust sampling by adjusting the degradation process. """
        batch_size = cond.shape[0]  # B
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        
        inf_weight = (self.noise_weights[self.max_step - 1] + self.info_weights[self.max_step - 1]).to(device)  # scalar
        x_s = inf_weight * torch.randn(data_dim, dtype=torch.float32, device=device)  # [B, N, S, A, 2]
        
        # Restore data from noise.
        for s in range(self.max_step - 1, -1, -1):  # reverse from t to 0
            x_0_hat = restore_fn(x_s, s * torch.ones(batch_size, dtype=torch.int64), cond)  # Restore x_0 from x_s
            if s > 0:
                # Adjusted robust sampling step: adjust degradation for smoother transition
                x_s = x_s - self.degrade_fn(x_0_hat, t=s * torch.ones(batch_size, dtype=torch.int64)) + \
                      self.degrade_fn(x_0_hat, t=(s - 1) * torch.ones(batch_size, dtype=torch.int64))  # Adjusted smoothing
        
        return x_0_hat

    def fast_sampling(self, restore_fn, cond, device):
        """ Fast sampling with a single pass of reverse diffusion. """
        batch_size = cond.shape[0]  # B
        batch_max = (self.max_step - 1) * torch.ones(batch_size, dtype=torch.int64)
        
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device)  # [B, N, S, A, 2]
        
        inf_weight = (self.noise_weights[self.max_step - 1] + self.info_weights[self.max_step - 1]).to(device)  # scalar
        x_s = inf_weight * noise  # [B, N, S, A, 2]
        
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)  # Restore x_0 from x_s
        return x_0_hat

    def native_sampling(self, restore_fn, data, cond, device):
        """ Native sampling that uses given data and its degradation to restore the data. """
        batch_size = cond.shape[0]
        batch_max = (self.max_step - 1) * torch.ones(batch_size, dtype=torch.int64)
        
        x_s = self.degrade_fn(data, batch_max).to(device)  # Degrade the input data
        x_0_hat = restore_fn(x_s, batch_max, cond)  # Restore x_0 from x_s
        
        return x_0_hat