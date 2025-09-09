import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConditionalUNet(nn.Module):
    def __init__(self, params):
        super(ConditionalUNet, self).__init__()
        self.task_id = params.task_id
        
        if self.task_id == 4:  # 流量预测任务
            self._build_traffic_unet(params)
        else:
            self._build_default_unet(params)
    
    def _build_traffic_unet(self, params):
        """为流量预测任务构建UNet - 修复版本"""
        self.n_apps = 20  # 应用数量
        self.embedding_dim = 128  # 环境嵌入维度
        self.seq_len = getattr(params, 'seq_len', 24)
        self.pred_len = getattr(params, 'pred_len', 12)
        
        # 数据通道
        self.data_channels = self.n_apps
        self.cond_channels = self.n_apps + self.embedding_dim
        
        # 时间嵌入
        self.time_embed_dim = 128
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim // 2, self.time_embed_dim),
        )
        
        # 🔧 修复：条件处理网络 - 适应1D时间序列
        self.cond_proj = nn.Sequential(
            nn.Conv1d(self.cond_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(self.pred_len)  # 调整到预测长度
        )
        
        # 🔧 修复：使用1D卷积而不是2D卷积
        # 输入维度：(B, n_apps + 32, pred_len)
        input_channels = self.data_channels + 32  # 32来自条件投影
        
        # 编码器 - 使用1D卷积
        self.down_blocks = nn.ModuleList([
            DownBlock1D(input_channels, 64, self.time_embed_dim),
            DownBlock1D(64, 128, self.time_embed_dim),
            DownBlock1D(128, 256, self.time_embed_dim),
        ])
        
        # 中间块
        self.mid_block = MiddleBlock1D(256, self.time_embed_dim)
        
        # 解码器
        self.up_blocks = nn.ModuleList([
            UpBlock1D(256 + 256, 128, self.time_embed_dim),
            UpBlock1D(128 + 128, 64, self.time_embed_dim),
            UpBlock1D(64 + 64, 32, self.time_embed_dim),
        ])
        
        # 输出层
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, self.data_channels, 3, padding=1),
        )
    
    def _build_default_unet(self, params):
        """默认UNet结构"""
        self.time_embed_dim = 128
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim // 2, self.time_embed_dim),
        )
        
        self.conv_in = nn.Conv2d(1, 64, 3, padding=1)
        self.conv_mid = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_out = nn.Conv2d(128, 1, 3, padding=1)
    
    def forward(self, x, t, cond=None):
        if self.task_id == 4:
            return self._forward_traffic(x, t, cond)
        else:
            return self._forward_default(x, t, cond)
    
    def _forward_traffic(self, x, t, cond):
        """🔧 修复：流量预测的前向传播 - 1D版本"""
        B, C, T, _ = x.shape  # (B, n_apps, pred_len, 1)
        
        # 🔧 将4D数据转换为3D：(B, n_apps, pred_len, 1) -> (B, n_apps, pred_len)
        x = x.squeeze(-1)  # (B, n_apps, pred_len)
        
        # 时间嵌入
        t_embed = self.time_embed(t.float().unsqueeze(-1))  # (B, time_embed_dim)
        
        # 处理条件
        if cond is not None:
            # cond: (B, n_apps + embedding_dim, seq_len)
            cond_feat = self.cond_proj(cond)  # (B, 32, pred_len)
            
            # 拼接数据和条件
            x = torch.cat([x, cond_feat], dim=1)  # (B, n_apps + 32, pred_len)
        
        # 🔧 UNet前向传播 - 1D版本
        skip_connections = []
        h = x
        
        # 编码器
        for down_block in self.down_blocks:
            h = down_block(h, t_embed)
            skip_connections.append(h)
            # 🔧 修复：使用1D池化，并确保不会变成0
            if h.shape[2] > 2:  # 只有当长度>2时才池化
                h = F.avg_pool1d(h, 2)
        
        # 中间块
        h = self.mid_block(h, t_embed)
        
        # 解码器
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            # 🔧 修复：1D插值
            if h.shape[2] != skip.shape[2]:
                h = F.interpolate(h, size=skip.shape[2], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_embed)
        
        # 输出
        out = self.out_conv(h)  # (B, n_apps, pred_len)
        
        # 🔧 转换回4D格式以匹配输入
        out = out.unsqueeze(-1)  # (B, n_apps, pred_len, 1)
        
        return out
    
    def _forward_default(self, x, t, cond=None):
        """默认前向传播"""
        t_embed = self.time_embed(t.float().unsqueeze(-1))
        
        h = self.conv_in(x)
        h = h + t_embed.unsqueeze(-1).unsqueeze(-1)
        h = F.relu(h)
        h = self.conv_mid(h)
        h = F.relu(h)
        out = self.conv_out(h)
        
        return out


# 🔧 新增：1D版本的网络块
class DownBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.activation = nn.SiLU()
    
    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(t_embed)  # (B, out_channels)
        h = h + time_emb.unsqueeze(-1)  # 广播到 (B, out_channels, T)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.activation = nn.SiLU()
    
    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(t_embed)
        h = h + time_emb.unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h


class MiddleBlock1D(nn.Module):
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, channels)
        
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        
        self.activation = nn.SiLU()
    
    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(t_embed)
        h = h + time_emb.unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h


# 保留原有的2D块以兼容其他任务
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.activation = nn.SiLU()
    
    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        time_emb = self.time_mlp(t_embed)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.activation = nn.SiLU()
    
    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        time_emb = self.time_mlp(t_embed)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h


class MiddleBlock(nn.Module):
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, channels)
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        
        self.activation = nn.SiLU()
    
    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        time_emb = self.time_mlp(t_embed)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h
