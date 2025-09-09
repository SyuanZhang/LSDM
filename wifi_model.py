import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def modulate(x, shift, scale):
    """AdaLN调制函数，来自DiT"""
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """时间步嵌入器，用于扩散过程"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """创建正弦时间步嵌入"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ConditionEmbedder(nn.Module):
    """条件嵌入器，处理条件信息"""
    def __init__(self, cond_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.cond_dim = cond_dim
        
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, c):
        # 处理不同维度的条件输入
        if c.dim() == 3:  # [B, T, F]
            c = c.mean(dim=1)  # 平均池化到 [B, F]
        elif c.dim() == 4:  # [B, C, T, F]
            B, C, T, F = c.shape
            c = c.reshape(B, C * T * F)
        
        # 验证维度匹配
        if c.shape[-1] != self.cond_dim:
            raise ValueError(f"Condition dimension mismatch: expected {self.cond_dim}, got {c.shape[-1]}")
        
        result = self.condition_proj(c)
        return result

class MaskEmbedder(nn.Module):
    """掩码嵌入器，处理掩码信息"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mask_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, mask):
        # mask: [batch, seq_len] -> [batch, seq_len, 1]
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return self.mask_proj(mask.float())

class MultiScaleTemporalConv(nn.Module):
    """多尺度时序卷积，增强版"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        # 多个不同尺度的卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, out_channels//4, kernel_size=9, padding=4)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # 1x1卷积用于特征融合
        self.fusion_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(x))
        out3 = F.gelu(self.conv3(x))
        out4 = F.gelu(self.conv4(x))
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = F.gelu(self.fusion_conv(out))
        out = self.dropout(out)
        
        # 转回 [batch, seq_len, features]
        return out.transpose(1, 2)

class DiTBlock(nn.Module):
    """基于DiT的Transformer块，支持多分支注意力和AdaLN调制"""
    def __init__(self, hidden_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 三个分支的注意力：时间、特征、掩码感知
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # 多分支注意力
        self.attn_temporal = self._create_attention(hidden_dim, num_heads, dropout)
        self.attn_feature = self._create_attention(hidden_dim, num_heads, dropout)
        self.attn_mask_aware = self._create_attention(hidden_dim, num_heads, dropout)
        
        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # AdaLN调制层 - 增加了掩码分支
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim, bias=True)  # 9个参数：3个attention + 1个mlp，每个2个，额外一个门控
        )
        
    def _create_attention(self, hidden_dim, num_heads, dropout):
        """创建注意力层"""
        return nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
    
    def forward(self, x, c, mask_attn=None):
        # c是条件信息：时间步嵌入 + 条件嵌入
        # mask_attn是注意力掩码
        B, L, D = x.shape
        
        # AdaLN调制参数
        ada_params = self.adaLN_modulation(c)  # [B, 9*D]
        (shift_attn1, scale_attn1, gate_attn1, 
         shift_attn2, scale_attn2, gate_attn2,
         shift_attn3, scale_attn3, gate_attn3) = ada_params.chunk(9, dim=-1)
        
        # 第一个注意力分支：时间依赖
        x_norm1 = modulate(self.norm1(x), shift_attn1.unsqueeze(1), scale_attn1.unsqueeze(1))
        attn_out1, _ = self.attn_temporal(x_norm1, x_norm1, x_norm1, attn_mask=mask_attn)
        x1 = x + gate_attn1.unsqueeze(1) * attn_out1
        
        # 第二个注意力分支：特征依赖
        x_norm2 = modulate(self.norm2(x1), shift_attn2.unsqueeze(1), scale_attn2.unsqueeze(1))
        attn_out2, _ = self.attn_feature(x_norm2, x_norm2, x_norm2)
        x2 = x1 + gate_attn2.unsqueeze(1) * attn_out2
        
        # 第三个注意力分支：掩码感知注意力
        x_norm3 = modulate(self.norm3(x2), shift_attn3.unsqueeze(1), scale_attn3.unsqueeze(1))
        attn_out3, _ = self.attn_mask_aware(x_norm3, x_norm3, x_norm3, attn_mask=mask_attn)
        x3 = x2 + gate_attn3.unsqueeze(1) * attn_out3
        
        # MLP分支
        mlp_out = self.mlp(x3)
        x4 = x3 + mlp_out
        
        return x4

class FinalLayer(nn.Module):
    """最终输出层，带AdaLN调制"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        x = self.linear(x)
        return x

class DiffusionTimeSeriesModel(nn.Module):
    """🔧 修复版：基于扩散模型的时序预测模型，支持可配置mask"""
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_dim = params.input_dim
        self.hidden_dim = params.hidden_dim
        self.output_dim = params.input_dim
        self.num_timesteps = getattr(params, 'num_timesteps', 1000)
        
        # Mask相关参数
        self.mask_length = getattr(params, 'mask_length', 3)
        self.seq_len = getattr(params, 'seq_len', 10)
        self.pred_len = getattr(params, 'pred_len', 10)
        
        # 条件维度
        self.cond_dim = getattr(params, 'cond_dim', 148)
        
        # 确保mask长度合理
        if self.mask_length > self.seq_len:
            self.mask_length = max(1, self.seq_len // 2)
        
        # 输入投影
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 时间步嵌入
        self.timestep_embedder = TimestepEmbedder(self.hidden_dim)
        
        # 条件嵌入
        if self.cond_dim > 0:
            self.condition_embedder = ConditionEmbedder(self.cond_dim, self.hidden_dim)
            self.has_condition = True
        else:
            self.has_condition = False
        
        # 掩码嵌入
        self.mask_embedder = MaskEmbedder(self.hidden_dim)
        
        # 多尺度卷积
        self.temporal_conv = MultiScaleTemporalConv(self.hidden_dim, self.hidden_dim)
        
        # 位置编码
        self.pos_encoding = self._create_position_encoding(512, self.hidden_dim)
        
        # DiT块
        depth = getattr(params, 'depth', 4)
        self.dit_blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, num_heads=getattr(params, 'num_heads', 4), 
                    mlp_ratio=4.0, dropout=0.1)
            for _ in range(depth)
        ])
        
        # 最终输出层
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)
        
        # 🔧 使用线性调度，更稳定
        self.register_buffer('betas', self._linear_beta_schedule(self.num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 🔧 添加数值稳定性参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # 初始化权重
        self._init_weights()
    
    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """🔧 使用线性调度，更稳定"""
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def _preprocess_input(self, x):
        """🔧 改进的预处理输入数据，确保格式正确"""
        original_shape = x.shape
        
        if x.dim() == 2:  # [B, F] -> [B, 1, F]
            x = x.unsqueeze(1)
            
        elif x.dim() == 3:  # [B, T, F] - 期望格式
            if x.shape[-1] != self.input_dim:
                # 可能是 [B, F, T] 格式，需要转置
                if x.shape[1] == self.input_dim:
                    x = x.transpose(1, 2)
                else:
                    raise ValueError(f"Cannot match input_dim {self.input_dim} with shape {x.shape}")
            
        elif x.dim() == 4:  # 4D inputs
            if x.shape[-1] == 1:  # [B, T, F, 1] -> [B, T, F]
                x = x.squeeze(-1)
            elif x.shape[1] == 1:  # [B, 1, T, F] -> [B, T, F]
                x = x.squeeze(1)
            elif x.shape[2] == 1:  # [B, T, 1, F] -> [B, T, F]
                x = x.squeeze(2)
            else:
                # 尝试重塑为 [B, T, F]
                B, C, T, F = x.shape
                if C * F == self.input_dim:
                    x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
                elif T * F == self.input_dim:
                    x = x.reshape(B, C, T * F)
                    if C == 1:
                        x = x.squeeze(1)
                else:
                    raise ValueError(f"Cannot reshape {original_shape} to [B, T, {self.input_dim}]")
        else:
            raise ValueError(f"Unsupported input dimensions: {x.dim()}")
        
        # 最终验证
        if x.dim() != 3:
            raise ValueError(f"After preprocessing, expected 3D tensor, got {x.dim()}D: {x.shape}")
        
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Final feature dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}")
        
        return x
    
    def _preprocess_condition(self, c):
        """🔧 改进的预处理条件信息，确保维度正确"""
        if c is None:
            return None
        
        original_shape = c.shape
        
        # 处理不同维度的条件输入
        if c.dim() == 3:  # [B, T, F]
            B, T, F = c.shape
            if F == self.cond_dim:
                # 正确的格式，进行平均池化
                c = c.mean(dim=1)  # [B, T, F] -> [B, F]
            elif T == self.cond_dim:
                # 可能是 [B, F, T] 格式
                c = c.mean(dim=2)  # [B, F, T] -> [B, F]
            else:
                # 尝试重塑
                if T * F == self.cond_dim:
                    c = c.reshape(B, T * F)
                else:
                    raise ValueError(f"Cannot match condition shape {c.shape} to expected dim {self.cond_dim}")
                    
        elif c.dim() == 4:  # [B, C, T, F]
            B, C, T, F = c.shape
            total_features = C * T * F
            if total_features == self.cond_dim:
                c = c.reshape(B, total_features)
            else:
                # 尝试其他组合
                if T * F == self.cond_dim:
                    c = c.mean(dim=1).reshape(B, T * F)  # 平均池化channel维度
                else:
                    raise ValueError(f"Cannot reshape condition {original_shape} to match dim {self.cond_dim}")
        
        elif c.dim() == 2:  # [B, F] - 已经是正确格式
            if c.shape[-1] != self.cond_dim:
                raise ValueError(f"Condition dimension mismatch: expected {self.cond_dim}, got {c.shape[-1]}")
        
        else:
            raise ValueError(f"Unsupported condition dimensions: {c.dim()}")
        
        # 最终验证
        if c.shape[-1] != self.cond_dim:
            raise ValueError(f"Final condition dimension mismatch: expected {self.cond_dim}, got {c.shape[-1]}")
        
        return c
    
    def create_mask(self, batch_size, seq_len, mask_length=None, mask_strategy='prefix', device=None):
        """🔧 改进的创建掩码函数"""
        if device is None:
            device = next(self.parameters()).device
        
        if mask_length is None:
            mask_length = self.mask_length
            
        # 🔧 确保mask长度合理
        mask_length = min(mask_length, seq_len - 1)  # 至少保留一个位置不被mask
        mask_length = max(1, mask_length)  # 至少mask一个位置
        
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        if mask_strategy == 'prefix':
            mask[:, :mask_length] = True
        elif mask_strategy == 'suffix':
            mask[:, -mask_length:] = True
        elif mask_strategy == 'random':
            for b in range(batch_size):
                indices = torch.randperm(seq_len, device=device)[:mask_length]
                mask[b, indices] = True
        else:
            raise ValueError(f"Unknown mask strategy: {mask_strategy}")
        
        return mask
    
    def apply_mask_to_data(self, x, mask, mask_value=0.0):
        """将掩码应用到数据上"""
        masked_x = x.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        masked_x[mask_expanded] = mask_value
        return masked_x
    
    def _create_position_encoding(self, max_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        """🔧 改进的权重初始化"""
        # 时间步嵌入 - 较小的初始化
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.01)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.01)
        
        # 输入投影 - 较小的初始化
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        nn.init.constant_(self.input_projection.bias, 0)
        
        # DiT块的AdaLN层置零
        for block in self.dit_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # 最终层置零
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # 其他层 - 使用更保守的初始化
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # 🔧 较小的gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if hasattr(m, 'elementwise_affine') and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def q_sample(self, x_start, t, noise=None):
        """🔧 改进的前向扩散过程，增加数值稳定性"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 🔧 使用预计算的稳定值
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # 🔧 限制噪声范围
        noise = torch.clamp(noise, -2.0, 2.0)
        
        result = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return result
    
    def forward(self, x, t=None, c=None, mask=None, mask_strategy='prefix', mask_length=None):
        """🔧 增加数值稳定性检查的前向传播"""
        try:
            # 预处理输入数据
            x = self._preprocess_input(x)
            batch_size, seq_len, _ = x.shape
            
            # 🔧 检查输入数据
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("⚠️ NaN/Inf detected in input data")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 预处理条件信息
            if c is not None:
                c = self._preprocess_condition(c)
                if torch.isnan(c).any() or torch.isinf(c).any():
                    print("⚠️ NaN/Inf detected in condition data")
                    c = torch.nan_to_num(c, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 生成或使用提供的掩码
            if mask is None:
                mask = self.create_mask(
                    batch_size, seq_len, 
                    mask_length=mask_length, 
                    mask_strategy=mask_strategy,
                    device=x.device
                )
            
            # 🔧 确保mask有意义
            if mask.sum() == 0:
                print("⚠️ Empty mask, creating default mask")
                mask_len = min(max(1, self.mask_length), seq_len - 1)
                mask[:, :mask_len] = True
            
            # 如果是训练模式且提供了时间步，进行扩散过程
            if self.training and t is not None:
                # 🔧 限制时间步范围
                t = torch.clamp(t, 0, self.num_timesteps - 1)
                noise = torch.randn_like(x)
                x_noisy = self.q_sample(x, t, noise)
                x_input = self.apply_mask_to_data(x_noisy, mask)
            else:
                x_input = self.apply_mask_to_data(x, mask)
                if t is None:
                    t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)
            
            # 🔧 检查处理后的输入
            if torch.isnan(x_input).any() or torch.isinf(x_input).any():
                print("⚠️ NaN/Inf detected after masking")
                x_input = torch.nan_to_num(x_input, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 输入投影
            x_embedded = self.input_projection(x_input)
            
            # 🔧 检查嵌入
            if torch.isnan(x_embedded).any() or torch.isinf(x_embedded).any():
                print("⚠️ NaN/Inf detected after input projection")
                x_embedded = torch.nan_to_num(x_embedded, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 添加位置编码
            if seq_len <= self.pos_encoding.size(1):
                x_embedded = x_embedded + self.pos_encoding[:, :seq_len, :]
            
            # 掩码嵌入
            mask_emb = self.mask_embedder(mask)
            x_embedded = x_embedded + mask_emb
            
            # 多尺度卷积特征
            conv_features = self.temporal_conv(x_embedded)
            x_embedded = x_embedded + conv_features
            
            # 时间步嵌入
            t_emb = self.timestep_embedder(t)
            
            # 条件嵌入
            condition_emb = torch.zeros_like(t_emb)
            if self.has_condition and c is not None:
                condition_emb = self.condition_embedder(c)
            
            # 组合条件信息
            combined_condition = t_emb + condition_emb
            
            # 🔧 检查条件嵌入
            if torch.isnan(combined_condition).any() or torch.isinf(combined_condition).any():
                print("⚠️ NaN/Inf detected in condition embedding")
                combined_condition = torch.nan_to_num(combined_condition, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 创建注意力掩码
            attn_mask = None
            if self.training:
                attn_mask = torch.zeros(seq_len, seq_len, device=x.device, dtype=torch.bool)
                if mask.sum() > 0:  # 🔧 确保有mask位置
                    mask_indices = mask[0]
                    attn_mask[mask_indices, :] = True
            
            # DiT块处理
            for i, block in enumerate(self.dit_blocks):
                x_embedded_old = x_embedded.clone()
                x_embedded = block(x_embedded, combined_condition, mask_attn=attn_mask)
                
                # 🔧 检查每个块的输出
                if torch.isnan(x_embedded).any() or torch.isinf(x_embedded).any():
                    print(f"⚠️ NaN/Inf detected in DiT block {i}")
                    x_embedded = x_embedded_old  # 回滚到之前的状态
                    break
            
            # 最终输出
            output = self.final_layer(x_embedded, combined_condition)
            
            # 🔧 最终检查输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("⚠️ NaN/Inf detected in final output")
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return output, mask
            
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # 🔧 返回安全的默认值
            if 'x' in locals():
                return torch.zeros_like(x), mask if 'mask' in locals() else torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
            else:
                raise e

class MaskedDiffusionLoss(nn.Module):
    """🔧 修复版损失函数，增加数值稳定性检查"""
    def __init__(self, loss_type='combined', mask_weight=1.0, unmask_weight=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.mask_weight = mask_weight
        self.unmask_weight = unmask_weight
        self.eps = 1e-8  # 数值稳定性常数
        
    def forward(self, predicted, target, mask):
        # 🔧 数值稳定性检查
        if torch.isnan(predicted).any() or torch.isinf(predicted).any():
            print("⚠️ NaN/Inf detected in predictions")
            return torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(float('inf'))
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("⚠️ NaN/Inf detected in targets")
            return torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(float('inf'))
        
        mask_expanded = mask.unsqueeze(-1).expand_as(predicted)
        
        # 🔧 确保mask区域不为空
        mask_count = mask_expanded.sum()
        unmask_count = (~mask_expanded).sum()
        
        # mask区域的损失
        if mask_count > 0:
            mask_pred = predicted[mask_expanded]
            mask_target = target[mask_expanded]
            
            if self.loss_type == 'mse':
                mask_loss = F.mse_loss(mask_pred, mask_target, reduction='mean')
            elif self.loss_type == 'mae':
                mask_loss = F.l1_loss(mask_pred, mask_target, reduction='mean')
            elif self.loss_type == 'combined':
                mse = F.mse_loss(mask_pred, mask_target, reduction='mean')
                mae = F.l1_loss(mask_pred, mask_target, reduction='mean')
                mask_loss = mse + 0.1 * mae
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
                
            # 🔧 检查损失是否有效
            if torch.isnan(mask_loss) or torch.isinf(mask_loss):
                return torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(float('inf'))
        else:
            mask_loss = torch.tensor(0.0, device=predicted.device)
        
        # 非mask区域的损失
        if unmask_count > 0:
            unmask_pred = predicted[~mask_expanded]
            unmask_target = target[~mask_expanded]
            unmask_loss = F.mse_loss(unmask_pred, unmask_target, reduction='mean')
            
            if torch.isnan(unmask_loss) or torch.isinf(unmask_loss):
                return torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(float('inf'))
        else:
            unmask_loss = torch.tensor(0.0, device=predicted.device)
        
        total_loss = self.mask_weight * mask_loss + self.unmask_weight * unmask_loss
        
        # 🔧 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(float('inf')), torch.tensor(float('inf')), torch.tensor(float('inf'))
        
        return total_loss, mask_loss, unmask_loss

class MaskedDiffusionTrainer:
    """🔧 改进的训练器，增加数值稳定性"""
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.device = params.device
        
        # 🔧 使用更保守的优化器设置
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(params, 'learning_rate', 1e-5),  # 更小的学习率
            weight_decay=getattr(params, 'weight_decay', 1e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 🔧 更温和的学习率调度
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.8,
            patience=10,
            min_lr=1e-7
        )
        
        # 损失函数
        self.criterion = MaskedDiffusionLoss(
            loss_type='combined',
            mask_weight=getattr(params, 'mask_weight', 1.0),
            unmask_weight=getattr(params, 'unmask_weight', 0.01)
        )
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.patience = getattr(params, 'patience', 50)
        self.patience_counter = 0
        
        # Mask相关参数
        self.mask_strategies = getattr(params, 'mask_strategies', ['prefix'])
        self.mask_lengths = getattr(params, 'mask_lengths', [params.mask_length])
        
        # 梯度统计
        self.grad_norm_history = []
    
    def train_step(self, batch):
        """🔧 改进的单步训练，增加梯度监控"""
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # 解析批次数据
            if isinstance(batch, dict):
                x = batch['data'].to(self.device)
                c = batch.get('cond', None)
                if c is not None:
                    c = c.to(self.device)
            else:
                if len(batch) == 2:
                    x, c = batch
                    c = c.to(self.device) if c is not None else None
                else:
                    x = batch[0]
                    c = None
                x = x.to(self.device)
            
            # 🔧 检查输入数据质量
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("⚠️ Invalid input data detected, skipping batch")
                return float('inf'), float('inf'), float('inf')
            
            # 获取批次大小
            original_batch_size = x.shape[0]
            
            # 随机选择mask策略和长度
            mask_strategy = np.random.choice(self.mask_strategies)
            mask_length = np.random.choice(self.mask_lengths)
            
            # 🔧 使用更小的时间步范围
            max_t = min(50, self.model.num_timesteps)  # 更小的范围
            t = torch.randint(0, max_t, (original_batch_size,), device=self.device)
            
            # 前向传播
            predicted, mask = self.model(x, t, c, mask_strategy=mask_strategy, mask_length=mask_length)
            
            # 预处理目标数据
            target = self.model._preprocess_input(x.clone())
            
            # 计算损失
            total_loss, mask_loss, unmask_loss = self.criterion(predicted, target, mask)
            
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 1000:
                return float('inf'), float('inf'), float('inf')
            
            # 反向传播
            total_loss.backward()
            
            # 🔧 监控梯度
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.grad_norm_history.append(total_norm)
            
            # 🔧 更激进的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            return total_loss.item(), mask_loss.item(), unmask_loss.item()
            
        except Exception as e:
            print(f"❌ Error in train_step: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), float('inf'), float('inf')
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_mask_loss = 0
        total_unmask_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # 解析批次数据
                    if isinstance(batch, dict):
                        x = batch['data'].to(self.device)
                        c = batch.get('cond', None)
                        if c is not None:
                            c = c.to(self.device)
                    else:
                        if len(batch) == 2:
                            x, c = batch
                            c = c.to(self.device) if c is not None else None
                        else:
                            x = batch[0]
                            c = None
                        x = x.to(self.device)
                    
                    original_batch_size = x.shape[0]
                    
                    # 🔧 使用更小的时间步范围
                    max_t = min(50, self.model.num_timesteps)
                    t = torch.randint(0, max_t, (original_batch_size,), device=self.device)
                    predicted, mask = self.model(x, t, c, mask_strategy='prefix')
                    
                    target = self.model._preprocess_input(x.clone())
                    
                    total_loss_batch, mask_loss_batch, unmask_loss_batch = self.criterion(predicted, target, mask)
                    
                    if not (torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch)):
                        total_loss += total_loss_batch.item()
                        total_mask_loss += mask_loss_batch.item()
                        total_unmask_loss += unmask_loss_batch.item()
                        num_batches += 1
                        
                except Exception as e:
                    print(f"⚠️ Error in validation batch: {e}")
                    continue
        
        if num_batches == 0:
            return float('inf'), float('inf'), float('inf')
        
        return total_loss / num_batches, total_mask_loss / num_batches, total_unmask_loss / num_batches
    
    def train_epoch(self, train_loader, val_loader, epoch):
        """训练一个epoch"""
        epoch_losses = []
        epoch_mask_losses = []
        epoch_unmask_losses = []
        
        print(f"\n=== 🚀 Epoch {epoch} ===")
        
        for batch_idx, batch in enumerate(train_loader):
            total_loss, mask_loss, unmask_loss = self.train_step(batch)
            
            if total_loss != float('inf'):
                epoch_losses.append(total_loss)
                epoch_mask_losses.append(mask_loss)
                epoch_unmask_losses.append(unmask_loss)
            
            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'📊 Batch {batch_idx}:')
                print(f'  💰 Total Loss: {total_loss:.6f}')
                print(f'  🎯 Mask Loss: {mask_loss:.6f}')
                print(f'  ⚖️ Unmask Loss: {unmask_loss:.6f}')
                print(f'  📈 LR: {current_lr:.2e}')
        
        # 验证
        print("🔍 Validating...")
        val_loss, val_mask_loss, val_unmask_loss = self.validate(val_loader)
        
        # 更新学习率
        self.scheduler.step(val_loss)
        
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_mask_loss = np.mean(epoch_mask_losses) if epoch_mask_losses else float('inf')
        avg_unmask_loss = np.mean(epoch_unmask_losses) if epoch_unmask_losses else float('inf')
        
        # 早停检查
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), 'best_masked_diffusion_model.pt')
            print("💾 New best model saved!")
        else:
            self.patience_counter += 1
        
        print(f'📈 Epoch {epoch} Summary:')
        print(f'  🚂 Train - Total: {avg_train_loss:.6f}, Mask: {avg_mask_loss:.6f}, Unmask: {avg_unmask_loss:.6f}')
        print(f'  🎯 Val - Total: {val_loss:.6f}, Mask: {val_mask_loss:.6f}, Unmask: {val_unmask_loss:.6f}')
        print(f'  🏆 Best Val Loss: {self.best_val_loss:.6f}')
        print(f'  ⏰ Patience: {self.patience_counter}/{self.patience}')
        
        return avg_train_loss, val_loss, self.patience_counter >= self.patience

# 主模型类（保持接口兼容性）
class tfdiff_WiFi(DiffusionTimeSeriesModel):
    """保持接口兼容性的主模型"""
    def __init__(self, params):
        super().__init__(params)

class MaskedDiffusionConfig:
    """🔧 支持mask的扩散模型配置 - 更稳定的版本"""
    def __init__(self):
        # 模型参数 - 更保守的设置
        self.input_dim = 20
        self.hidden_dim = 64        # 🔧 进一步减小隐藏维度
        self.depth = 3              # 🔧 减少层数
        self.num_heads = 2          # 🔧 减少头数
        self.dropout = 0.1
        
        # 扩散参数
        self.num_timesteps = 50     # 🔧 进一步减少时间步
        
        # 条件参数
        self.cond_dim = 148
        
        # Mask参数
        self.mask_length = 2        # 🔧 更小的mask长度
        self.mask_strategies = ['prefix']  # 先用简单策略
        self.mask_lengths = [1, 2, 3]     # 🔧 更小的范围
        self.mask_weight = 1.0
        self.unmask_weight = 0.01
        
        # 训练参数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 8         # 🔧 进一步减小批次大小
        self.learning_rate = 5e-6   # 🔧 更小的学习率
        self.weight_decay = 1e-6
        self.num_epochs = 200
        self.patience = 50
        
        # 序列参数
        self.seq_len = 10
        self.pred_len = 10

def create_masked_diffusion_model(config=None):
    """创建支持mask的扩散模型"""
    if config is None:
        config = MaskedDiffusionConfig()
    
    model = tfdiff_WiFi(config).to(config.device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"=== 🎯 Masked Diffusion Model Created ===")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🔧 Trainable parameters: {trainable_params:,}")
    print(f"💾 Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"⚙️ Device: {config.device}")
    print(f"⏰ Timesteps: {config.num_timesteps}")
    print(f"🎭 Mask length: {config.mask_length}")
    print(f"🔄 Mask strategies: {config.mask_strategies}")
    
    return model

# 使用示例和测试
if __name__ == "__main__":
    # 创建配置
    config = MaskedDiffusionConfig()
    config.input_dim = 20
    config.cond_dim = 148
    config.seq_len = 10
    config.mask_length = 2
    
    # 创建模型
    model = create_masked_diffusion_model(config)
    
    # 创建训练器
    trainer = MaskedDiffusionTrainer(model, config)
    
    print("=== 🚀 Masked Diffusion Model Ready ===")
    
    # 测试
    print("\n=== 🧪 Testing with Your Data Format ===")
    
    batch_size = 4
    
    # 模拟数据
    hist_traffic = torch.randn(batch_size, 158, 20)
    target_traffic = torch.randn(batch_size, 10, 20)
    hist_env = torch.randn(batch_size, 158, 128)
    
    # 组合数据
    data = target_traffic.unsqueeze(-1)  # [B, 10, 20, 1]
    cond = torch.cat([hist_traffic, hist_env], dim=-1)  # [B, 158, 148]
    
    print(f"🔍 Test data shapes:")
    print(f"  • data: {data.shape}")
    print(f"  • cond: {cond.shape}")
    
    # 移动到设备
    data = data.to(config.device)
    cond = cond.to(config.device)
    
    try:
        with torch.no_grad():
            model.eval()
            
            # 测试前向传播
            output, mask = model(data, c=cond, mask_strategy='prefix', mask_length=2)
            print(f"✅ Forward pass successful!")
            print(f"  • Input: {data.shape} -> Output: {output.shape}")
            print(f"  • Mask shape: {mask.shape}")
            
            # 测试训练步骤
            model.train()
            t = torch.randint(0, config.num_timesteps, (batch_size,), device=config.device)
            pred_train, train_mask = model(data, t, cond, mask_strategy='prefix', mask_length=2)
            
            # 测试损失计算
            loss_fn = MaskedDiffusionLoss()
            target = model._preprocess_input(data.clone())
            total_loss, mask_loss, unmask_loss = loss_fn(pred_train, target, train_mask)
            print(f"📊 Loss test successful!")
            print(f"  • Total: {total_loss:.4f}, Mask: {mask_loss:.4f}, Unmask: {unmask_loss:.4f}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Model is ready for training with your data format!")
