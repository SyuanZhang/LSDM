import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from scipy import stats
import json

# 导入训练代码中的模块
from params import all_params
from wifi_model import tfdiff_WiFi
from model import ConditionalUNet
from diffusion import SignalDiffusion, GaussianDiffusion
from dataset import from_path

# 🔧 新增：导入修复版的扩散模型
try:
    from wifi_model import (
        DiffusionTimeSeriesModel, 
        MaskedDiffusionConfig, 
        create_masked_diffusion_model,
        MaskedDiffusionTrainer
    )
    MASKED_DIFFUSION_AVAILABLE = True
    print("✅ Masked diffusion model available")
except ImportError as e:
    print(f"⚠️ Masked diffusion model not available: {e}")
    MASKED_DIFFUSION_AVAILABLE = False

# 设置中文字体支持和绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def get_device(device_str='cuda'):
    """获取设备"""
    if device_str == 'cpu':
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        print("CUDA not available, falling back to CPU")
        return torch.device('cpu')

def seed_everything(seed=42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def detect_model_type(checkpoint_path):
    """🔧 增强的模型类型检测，支持更多模型类型"""
    print(f"Detecting model type from checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的检查点格式
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 检查关键的模型参数来判断模型类型
        keys = list(state_dict.keys())
        
        # 检查修复版扩散模型
        masked_diffusion_keys = [
            'timestep_embedder.mlp.0.weight',
            'condition_embedder.condition_proj.0.weight',
            'dit_blocks.0.adaLN_modulation.1.weight'
        ]
        
        # 如果有这些关键字，说明是ConditionalUNet
        conditional_unet_keys = ['time_embed.0.weight', 'down_blocks.0.time_mlp.weight', 'up_blocks.0.time_mlp.weight']
        
        # 如果有这些关键字，说明是tfdiff_WiFi
        tfdiff_wifi_keys = ['p_embed.projection.weight', 'blocks.0.norm1.weight', 'final_layer.norm.weight']
        
        has_masked_diffusion = any(key in keys for key in masked_diffusion_keys)
        has_conditional_unet = any(key in keys for key in conditional_unet_keys)
        has_tfdiff_wifi = any(key in keys for key in tfdiff_wifi_keys)
        
        if has_masked_diffusion:
            print("🔍 Detected model type: MaskedDiffusion")
            return 'MaskedDiffusion'
        elif has_conditional_unet:
            print("🔍 Detected model type: ConditionalUNet")
            return 'ConditionalUNet'
        elif has_tfdiff_wifi:
            print("🔍 Detected model type: tfdiff_WiFi")
            return 'tfdiff_WiFi'
        else:
            print("⚠️  Could not detect model type, defaulting to ConditionalUNet")
            print(f"Available keys (first 10): {keys[:10]}")
            return 'ConditionalUNet'
            
    except Exception as e:
        print(f"❌ Error detecting model type: {e}")
        print("Defaulting to ConditionalUNet")
        return 'ConditionalUNet'

def load_model_checkpoint(model, checkpoint_path, device):
    """🔧 改进的检查点加载，支持不完全匹配"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的检查点格式
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 处理 DistributedDataParallel 的权重
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
        else:
            new_state_dict[k] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Checkpoint loaded successfully")
    except RuntimeError as e:
        print(f"❌ Error loading checkpoint with strict=True: {e}")
        print("🔧 Trying to load with strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)} keys")
            print(f"First few missing keys: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)} keys")
            print(f"First few unexpected keys: {unexpected_keys[:5]}")
        print("✅ Checkpoint loaded with warnings")

def create_diffusion(params, device):
    """创建扩散模型"""
    if hasattr(params, 'signal_diffusion') and params.signal_diffusion:
        diffusion = SignalDiffusion(params)
    else:
        diffusion = GaussianDiffusion(params)
    return diffusion.to(device)

def compute_metrics(y_pred, y_true):
    """计算评估指标（修复JSON序列化问题）"""
    # 确保输入是numpy数组
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    
    # 🔧 处理NaN值
    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        print("⚠️ NaN values detected in predictions or targets")
        # 移除NaN值对应的位置
        valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        if valid_mask.sum() == 0:
            print("❌ All values are NaN!")
            return {
                'MSE': float('inf'),
                'MAE': float('inf'),
                'RMSE': float('inf'),
                'MAPE': float('inf'),
                'Correlation': 0.0,
                'R2_Score': 0.0
            }
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
    
    # 计算各种指标
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    # 避免除零错误
    y_true_nonzero = y_true + 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
    
    # 相关系数
    if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        correlation = np.corrcoef(y_pred.flatten(), y_true.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # R²决定系数
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    # 关键修改：将所有numpy类型转换为Python原生类型
    return {
        'MSE': float(mse),
        'MAE': float(mae), 
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'Correlation': float(correlation),
        'R2_Score': float(r2_score)
    }

def build_params(task_id, device_str='cuda', pred_len=20):
    """🔧 改进的参数构建函数，支持修复版扩散模型"""
    if task_id == 4:
        from params import AttrDict
        import numpy as np
        
        # 🔧 根据预测长度动态调整参数
        total_seq_len = 168  # 总序列长度
        input_seq_len = total_seq_len - pred_len  # 输入序列长度
        
        params = AttrDict(
            task_id=4,
            log_dir='./log/traffic',
            model_dir='./model/traffic_prediction',  
            data_dir=['./dataset/traffic'],
            traffic_path='traffic_data_new.npz',
            embedding_path='environment_embeddings.npz',
            max_iter=10000,
            batch_size=64,
            learning_rate=1e-4,
            max_grad_norm=0.5,
            inference_batch_size=16,
            robust_sampling=True,
            
            # 🔧 预测任务相关参数
            pred_len=pred_len,
            seq_len=input_seq_len,  
            input_seq_len=input_seq_len,
            total_seq_len=total_seq_len,
            
            sample_rate=20,
            input_dim=20,
            output_dim=20,
            extra_dim=[128],
            cond_dim=input_seq_len,  
            embed_dim=128,
            hidden_dim=128,
            num_heads=4,
            num_block=8,
            dropout=0.1,
            mlp_ratio=4.0,
            learn_tfdiff=False,
            max_step=1000,
            signal_diffusion=True,
            blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
            noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
            device=device_str,
            
            prediction_mode=True,  
            
            # 🔧 为修复版扩散模型添加参数
            mask_length=3,
            mask_strategies=['prefix', 'suffix', 'random'],
            mask_lengths=[1, 2, 3, 4, 5],
            mask_weight=1.0,
            unmask_weight=0.1,
            num_timesteps=1000,
            depth=4,
        )
        
        print(f"📊 Task {task_id} Configuration:")
        print(f"  • Total sequence length: {total_seq_len}")
        print(f"  • Input sequence length: {input_seq_len}")
        print(f"  • Prediction length: {pred_len}")
        print(f"  • Condition dimension: {input_seq_len}")
        
        return params
    else:
        params = all_params[task_id]
        # 为其他任务也添加预测模式支持
        if pred_len is not None and pred_len > 0:
            params.pred_len = pred_len
            params.prediction_mode = True
        return params

def evaluate_model(model, diffusion, dataloader, device, params, num_samples=None):
    """🔧 改进的模型评估函数，支持多种模型类型"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    total_samples = 0
    start_time = time.time()
    
    print("Starting evaluation...")
    
    # 🔧 检查模型类型
    model_type = type(model).__name__
    print(f"Model type detected: {model_type}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 将数据移到设备上
                if isinstance(batch, dict):
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(device)
                    
                    # 获取输入数据和目标
                    if 'data' in batch:
                        x = batch['data']
                        if x.dim() == 4 and x.size(-1) == 1:
                            target = x.squeeze(-1)  # [B, A, T]
                        else:
                            target = x
                    elif 'target_traffic' in batch:
                        target = batch['target_traffic']
                        x = target.unsqueeze(-1) if target.dim() == 3 else target
                    else:
                        print("Warning: Could not find data or target_traffic in batch")
                        continue
                    
                    # 条件信息
                    cond = batch.get('cond', None)
                    
                else:
                    # 如果batch不是字典，假设是tensor
                    x = batch.to(device)
                    target = x.squeeze(-1) if x.dim() == 4 else x
                    cond = None
                
                batch_size = target.size(0)
                
                # 🔧 根据模型类型进行推理
                if 'MaskedDiffusion' in model_type or 'DiffusionTimeSeriesModel' in model_type:
                    # 修复版扩散模型
                    if hasattr(model, 'num_timesteps'):
                        t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
                    else:
                        t = torch.randint(0, 50, (batch_size,), device=device)  # 默认值
                    
                    # 修复版模型的调用方式
                    pred, mask = model(x, t, cond)
                    
                elif 'ConditionalUNet' in model_type:
                    # ConditionalUNet模型
                    t = torch.randint(0, diffusion.max_step, (batch_size,), device=device)
                    if cond is not None:
                        pred = model(x, t, cond)
                    else:
                        pred = model(x, t)
                        
                else:
                    # tfdiff_WiFi模型
                    t = torch.randint(0, diffusion.max_step, (batch_size,), device=device)
                    if cond is not None:
                        pred = model(x, t, cond)
                    else:
                        pred = model(x, t)
                
                # 处理预测结果的维度
                if pred.dim() == 4 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                
                # 🔧 检查预测结果的有效性
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"⚠️ Invalid predictions in batch {batch_idx}, skipping")
                    continue
                
                # 收集结果
                all_predictions.append(pred.cpu())
                all_targets.append(target.cpu())
                
                total_samples += batch_size
                
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {batch_idx + 1} batches, {total_samples} samples, "
                          f"time: {elapsed:.2f}s")
                
                # 如果指定了样本数限制
                if num_samples is not None and total_samples >= num_samples:
                    break
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                # 跳过这个batch
                continue
    
    if not all_predictions:
        raise RuntimeError("No successful predictions were made!")
    
    # 合并所有预测和目标
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"Evaluation completed. Total samples: {total_samples}")
    print(f"Prediction shape: {all_predictions.shape}")
    print(f"Target shape: {all_targets.shape}")
    
    # 计算指标
    metrics = compute_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets

def create_comprehensive_visualizations(predictions, targets, metrics, output_dir, task_id):
    """创建全面的可视化图表"""
    print("Creating comprehensive visualizations...")
    
    # 确保输出目录存在
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 转换为numpy数组
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    print(f"Data shape - Predictions: {predictions.shape}, Targets: {targets.shape}")
    
    # 1. 预测效果总览
    create_prediction_overview(predictions, targets, metrics, vis_dir)
    
    # 2. 详细的时间序列预测对比
    create_detailed_time_series(predictions, targets, vis_dir)
    
    # 3. 预测精度分析
    create_accuracy_analysis(predictions, targets, vis_dir)
    
    # 4. 误差分析
    create_error_analysis(predictions, targets, vis_dir)
    
    # 5. 统计分布对比
    create_distribution_comparison(predictions, targets, vis_dir)
    
    # 6. 性能指标可视化
    create_metrics_dashboard(metrics, vis_dir)
    
    # 7. 如果是多维数据，创建热力图
    if predictions.ndim >= 3:
        create_multidimensional_analysis(predictions, targets, vis_dir)
    
    print(f"All visualizations saved to: {vis_dir}")

def create_prediction_overview(predictions, targets, metrics, output_dir):
    """创建预测效果总览图"""
    fig = plt.figure(figsize=(20, 12))
    
    # 🔧 修改：根据实际需要的子图数量调整网格大小
    n_time_series_examples = min(3, predictions.shape[0])  # 最多显示3个时间序列例子
    grid_rows = max(3, n_time_series_examples + 1)  # 至少3行，确保有足够空间
    
    gs = fig.add_gridspec(grid_rows, 4, hspace=0.3, wspace=0.3)
    
    # 1. 预测vs真实值散点图
    ax1 = fig.add_subplot(gs[0, 0:2])
    n_samples = min(10000, predictions.size)
    indices = np.random.choice(predictions.size, n_samples, replace=False)
    pred_flat = predictions.flatten()[indices]
    true_flat = targets.flatten()[indices]
    
    scatter = ax1.scatter(true_flat, pred_flat, alpha=0.6, s=2, c=pred_flat, cmap='viridis')
    min_val, max_val = min(true_flat.min(), pred_flat.min()), max(true_flat.max(), pred_flat.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Prediction vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Predicted Value')
    
    # 2. 相关性和R²显示
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.text(0.1, 0.8, f'Correlation: {metrics["Correlation"]:.4f}', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f'R² Score: {metrics["R2_Score"]:.4f}', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.4, f'RMSE: {metrics["RMSE"]:.4f}', fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.2, f'MAE: {metrics["MAE"]:.4f}', fontsize=14, transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Key Metrics')
    ax2.axis('off')
    
    # 3. 误差分布
    ax3 = fig.add_subplot(gs[0, 3])
    errors = (predictions - targets).flatten()
    ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.4f}')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 时间序列示例
    if predictions.ndim >= 2:
        for i in range(n_time_series_examples):
            # 🔧 修改：确保索引不会超出范围
            row_idx = i + 1
            if row_idx < grid_rows:  # 确保不超出网格范围
                ax = fig.add_subplot(gs[row_idx, :2])
                
                if predictions.ndim == 3:  # [samples, features, time]
                    # 🔧 改进：对于多维数据，显示所有特征的平均值
                    pred_series = predictions[i].mean(axis=0)
                    true_series = targets[i].mean(axis=0)
                    
                else:  # [samples, time]
                    pred_series = predictions[i]
                    true_series = targets[i]
                
                time_steps = np.arange(len(pred_series))
                ax.plot(time_steps, true_series, 'b-', linewidth=2, alpha=0.8, label='Ground Truth')
                ax.plot(time_steps, pred_series, 'r--', linewidth=2, alpha=0.8, label='Prediction')
                ax.fill_between(time_steps, true_series, pred_series, alpha=0.3, color='gray')
                
                ax.set_title(f'Sample {i+1} - Time Series Comparison')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    # 5. 整体误差趋势（如果是时间序列数据）
    if predictions.ndim >= 2 and predictions.shape[-1] > 1:
        # 🔧 修改：使用剩余的网格空间
        if grid_rows > 2:  # 确保有足够的行数
            ax5 = fig.add_subplot(gs[1:, 2:])
            
            # 计算每个时间步的平均绝对误差
            if predictions.ndim == 3:
                timestep_errors = np.abs(predictions - targets).mean(axis=(0, 1))
            else:
                timestep_errors = np.abs(predictions - targets).mean(axis=0)
            
            time_steps = np.arange(len(timestep_errors))
            ax5.plot(time_steps, timestep_errors, 'g-', linewidth=3, marker='o', markersize=4)
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Mean Absolute Error')
            ax5.set_title('Prediction Error by Time Step')
            ax5.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(time_steps) > 1:  # 确保有足够的数据点来拟合趋势线
                z = np.polyfit(time_steps, timestep_errors, 1)
                p = np.poly1d(z)
                ax5.plot(time_steps, p(time_steps), "r--", alpha=0.8, 
                        label=f'Trend (slope: {z[0]:.6f})')
                ax5.legend()
    
    plt.suptitle(f'Prediction Performance Overview - Shape {predictions.shape}', fontsize=16, y=0.98)
    plt.savefig(os.path.join(output_dir, 'prediction_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_time_series(predictions, targets, output_dir, n_examples=8):
    """创建详细的时间序列对比图"""
    if predictions.ndim < 2:
        return
    
    n_samples = min(predictions.shape[0], n_examples)
    fig, axes = plt.subplots((n_samples + 1) // 2, 2, figsize=(20, 5 * (n_samples + 1) // 2))
    
    # 🔧 确保axes是数组格式
    if n_samples <= 2:
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # 选择有代表性的样本（包括最好和最差的预测）
    sample_errors = []
    for i in range(min(50, predictions.shape[0])):  # 只计算前50个样本的误差，避免计算量过大
        if predictions.ndim == 3:
            error = np.mean(np.abs(predictions[i] - targets[i]))
        else:
            error = np.mean(np.abs(predictions[i] - targets[i]))
        sample_errors.append((error, i))
    
    sample_errors.sort()
    # 选择最好的几个和最差的几个，以及一些中等的
    best_indices = [idx for _, idx in sample_errors[:max(1, n_samples//3)]]
    worst_indices = [idx for _, idx in sample_errors[-max(1, n_samples//3):]]
    mid_start = len(sample_errors)//2 - max(1, n_samples//6)
    mid_end = len(sample_errors)//2 + max(1, n_samples//6)
    mid_indices = [idx for _, idx in sample_errors[mid_start:mid_end]]
    
    selected_indices = (best_indices + mid_indices + worst_indices)[:n_samples]
    
    for i, sample_idx in enumerate(selected_indices):
        ax = axes[i]
        
        if predictions.ndim == 3:  # [samples, features, time]
            # 显示所有特征的平均值
            pred_series = predictions[sample_idx].mean(axis=0)
            true_series = targets[sample_idx].mean(axis=0)
        else:  # [samples, time]
            pred_series = predictions[sample_idx]
            true_series = targets[sample_idx]
        
        time_steps = np.arange(len(pred_series))
        
        # 绘制预测和真实值
        ax.plot(time_steps, true_series, 'b-', linewidth=2.5, alpha=0.8, label='Ground Truth')
        ax.plot(time_steps, pred_series, 'r--', linewidth=2, alpha=0.9, label='Prediction')
        
        # 填充误差区域
        ax.fill_between(time_steps, true_series, pred_series, alpha=0.3, color='gray', label='Error')
        
        # 计算并显示该样本的误差
        mae = np.mean(np.abs(pred_series - true_series))
        rmse = np.sqrt(np.mean((pred_series - true_series) ** 2))
        
        ax.set_title(f'Sample {sample_idx + 1} (MAE: {mae:.4f}, RMSE: {rmse:.4f})')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(selected_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Detailed Time Series Prediction Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_accuracy_analysis(predictions, targets, output_dir):
    """创建预测精度分析"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. 绝对误差vs预测值
    pred_flat = predictions.flatten()
    abs_errors = np.abs(predictions.flatten() - targets.flatten())
    
    # 随机采样以避免过多的点
    n_samples = min(10000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_samples, replace=False)
    
    axes[0].scatter(pred_flat[indices], abs_errors[indices], alpha=0.5, s=1)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title('Absolute Error vs Predicted Values')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 相对误差分布
    relative_errors = np.abs((predictions.flatten() - targets.flatten()) / (targets.flatten() + 1e-8)) * 100
    relative_errors = relative_errors[relative_errors < 200]  # 移除异常值
    
    axes[1].hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Relative Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Relative Error Distribution')
    axes[1].axvline(np.median(relative_errors), color='red', linestyle='--', 
                   label=f'Median: {np.median(relative_errors):.2f}%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 误差累积分布函数
    errors_sorted = np.sort(abs_errors)
    cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
    
    axes[2].plot(errors_sorted, cdf, linewidth=2)
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Cumulative Probability')
    axes[2].set_title('Error Cumulative Distribution')
    axes[2].grid(True, alpha=0.3)
    
    # 4. 真实值vs绝对误差
    true_flat = targets.flatten()
    axes[3].scatter(true_flat[indices], abs_errors[indices], alpha=0.5, s=1)
    axes[3].set_xlabel('True Values')
    axes[3].set_ylabel('Absolute Error')
    axes[3].set_title('Absolute Error vs True Values')
    axes[3].grid(True, alpha=0.3)
    
    # 5. 预测值分布vs真实值分布
    axes[4].hist(true_flat, bins=50, alpha=0.7, label='True Values', color='blue', density=True)
    axes[4].hist(pred_flat, bins=50, alpha=0.7, label='Predictions', color='red', density=True)
    axes[4].set_xlabel('Value')
    axes[4].set_ylabel('Density')
    axes[4].set_title('Value Distributions')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # 6. 百分位数对比
    percentiles = np.arange(0, 101, 5)
    true_percentiles = np.percentile(true_flat, percentiles)
    pred_percentiles = np.percentile(pred_flat, percentiles)
    
    axes[5].plot(percentiles, true_percentiles, 'b-', linewidth=2, label='True Values', marker='o')
    axes[5].plot(percentiles, pred_percentiles, 'r--', linewidth=2, label='Predictions', marker='s')
    axes[5].set_xlabel('Percentile')
    axes[5].set_ylabel('Value')
    axes[5].set_title('Percentile Comparison')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Accuracy Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_error_analysis(predictions, targets, output_dir):
    """创建详细的误差分析"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    errors = predictions.flatten() - targets.flatten()
    
    # 1. 误差的正态性检验图
    axes[0].hist(errors, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 拟合正态分布
    mu, std = stats.norm.fit(errors)
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0].plot(x, p, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.4f}, σ={std:.4f})')
    
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Error Distribution with Normal Fit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Q-Q 图检验正态性
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot for Normality Check')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 误差的自相关分析（如果是时间序列）
    if predictions.ndim >= 2:
        # 选择一个样本进行自相关分析
        sample_errors = (predictions[0] - targets[0]).flatten() if predictions.ndim == 3 else predictions[0] - targets[0]
        if len(sample_errors) > 1:
            lags = range(1, min(20, len(sample_errors)))
            autocorrs = [np.corrcoef(sample_errors[:-lag], sample_errors[lag:])[0, 1] for lag in lags]
            
            axes[2].plot(lags, autocorrs, 'o-', linewidth=2)
            axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[2].set_xlabel('Lag')
            axes[2].set_ylabel('Autocorrelation')
            axes[2].set_title('Error Autocorrelation (Sample 1)')
            axes[2].grid(True, alpha=0.3)
    
    # 4. 误差统计摘要
    error_stats = {
        'Mean': np.mean(errors),
        'Std': np.std(errors),
        'Min': np.min(errors),
        'Max': np.max(errors),
        'Skewness': stats.skew(errors),
        'Kurtosis': stats.kurtosis(errors)
    }
    
    axes[3].axis('off')
    y_pos = 0.9
    axes[3].text(0.1, y_pos, 'Error Statistics:', fontsize=16, fontweight='bold', transform=axes[3].transAxes)
    for i, (stat, value) in enumerate(error_stats.items()):
        y_pos -= 0.12
        axes[3].text(0.1, y_pos, f'{stat}: {value:.6f}', fontsize=12, transform=axes[3].transAxes)
    
    plt.suptitle('Detailed Error Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_comparison(predictions, targets, output_dir):
    """创建预测值和真实值的分布对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    pred_flat = predictions.flatten()
    true_flat = targets.flatten()
    
    # 1. 直方图对比
    axes[0].hist(true_flat, bins=50, alpha=0.7, label='True Values', color='blue', density=True)
    axes[0].hist(pred_flat, bins=50, alpha=0.7, label='Predictions', color='red', density=True)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 箱线图对比
    data_to_plot = [true_flat, pred_flat]
    axes[1].boxplot(data_to_plot, labels=['True Values', 'Predictions'])
    axes[1].set_ylabel('Value')
    axes[1].set_title('Box Plot Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 小提琴图对比
    parts = axes[2].violinplot([true_flat, pred_flat], positions=[1, 2])
    for pc, color in zip(parts['bodies'], ['blue', 'red']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(['True Values', 'Predictions'])
    axes[2].set_ylabel('Value')
    axes[2].set_title('Violin Plot Comparison')
    axes[2].grid(True, alpha=0.3)
    
    # 4. 累积分布函数对比
    true_sorted = np.sort(true_flat)
    pred_sorted = np.sort(pred_flat)
    true_cdf = np.arange(1, len(true_sorted) + 1) / len(true_sorted)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
    
    axes[3].plot(true_sorted, true_cdf, 'b-', linewidth=2, label='True Values')
    axes[3].plot(pred_sorted, pred_cdf, 'r--', linewidth=2, label='Predictions')
    axes[3].set_xlabel('Value')
    axes[3].set_ylabel('Cumulative Probability')
    axes[3].set_title('Cumulative Distribution Comparison')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. 分布统计对比
    true_stats = {
        'Mean': np.mean(true_flat),
        'Std': np.std(true_flat),
        'Median': np.median(true_flat),
        'Skewness': stats.skew(true_flat),
        'Kurtosis': stats.kurtosis(true_flat)
    }
    
    pred_stats = {
        'Mean': np.mean(pred_flat),
        'Std': np.std(pred_flat),
        'Median': np.median(pred_flat),
        'Skewness': stats.skew(pred_flat),
        'Kurtosis': stats.kurtosis(pred_flat)
    }
    
    axes[4].axis('off')
    axes[4].text(0.05, 0.95, 'Distribution Statistics', fontsize=14, fontweight='bold', transform=axes[4].transAxes)
    axes[4].text(0.05, 0.85, 'True Values:', fontsize=12, fontweight='bold', color='blue', transform=axes[4].transAxes)
    axes[4].text(0.55, 0.85, 'Predictions:', fontsize=12, fontweight='bold', color='red', transform=axes[4].transAxes)
    
    y_pos = 0.75
    for stat in true_stats.keys():
        axes[4].text(0.05, y_pos, f'{stat}: {true_stats[stat]:.4f}', fontsize=10, transform=axes[4].transAxes)
        axes[4].text(0.55, y_pos, f'{stat}: {pred_stats[stat]:.4f}', fontsize=10, transform=axes[4].transAxes)
        y_pos -= 0.08
    
    # 6. 分位数-分位数图
    n_quantiles = 100
    quantiles = np.linspace(0, 100, n_quantiles)
    true_quantiles = np.percentile(true_flat, quantiles)
    pred_quantiles = np.percentile(pred_flat, quantiles)
    
    axes[5].scatter(true_quantiles, pred_quantiles, alpha=0.6)
    min_val = min(true_quantiles.min(), pred_quantiles.min())
    max_val = max(true_quantiles.max(), pred_quantiles.max())
    axes[5].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
    axes[5].set_xlabel('True Value Quantiles')
    axes[5].set_ylabel('Predicted Value Quantiles')
    axes[5].set_title('Quantile-Quantile Plot')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution Comparison Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_dashboard(metrics, output_dir):
    """创建性能指标仪表板"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # 颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
    
    # 为每个指标创建柱状图
    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
        if i < len(axes):
            ax = axes[i]
            bar = ax.bar([name], [value], alpha=0.8, color=colors[i], 
                        edgecolor='black', linewidth=2)
            ax.text(0, value/2, f'{value:.4f}', ha='center', va='center', 
                   fontsize=14, fontweight='bold')
            ax.set_title(f'{name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 为某些指标添加参考线
            if name == 'Correlation':
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect')
                ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Excellent')
                ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Good')
                ax.legend()
            elif name == 'R2_Score':
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect')
                ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good')
                ax.legend()
    
    # 隐藏多余的子图
    for i in range(len(metric_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Performance Metrics Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_multidimensional_analysis(predictions, targets, output_dir):
    """创建多维数据分析"""
    if predictions.ndim != 3:
        return
    
    n_samples, n_features, seq_len = predictions.shape
    
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 特征维度的平均误差
    feature_errors = np.mean(np.abs(predictions - targets), axis=(0, 2))
    
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(n_features), feature_errors, alpha=0.8)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error by Feature Dimension')
    ax1.grid(True, alpha=0.3)
    
    # 2. 时间步的平均误差
    timestep_errors = np.mean(np.abs(predictions - targets), axis=(0, 1))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(seq_len), timestep_errors, 'o-', linewidth=2, markersize=4)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Error by Time Step')
    ax2.grid(True, alpha=0.3)
    
    # 3. 特征维度热力图（前10个样本）
    ax3 = fig.add_subplot(gs[0, 2])
    sample_errors = np.abs(predictions[:10] - targets[:10])  # 前10个样本
    avg_sample_errors = np.mean(sample_errors, axis=2)  # 对时间维度求平均
    
    im = ax3.imshow(avg_sample_errors.T, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Feature Index')
    ax3.set_title('Error Heatmap (Samples vs Features)')
    plt.colorbar(im, ax=ax3)
    
    # 4. 时间-特征误差热力图
    ax4 = fig.add_subplot(gs[0, 3])
    avg_errors_time_feature = np.mean(np.abs(predictions - targets), axis=0)
    
    im = ax4.imshow(avg_errors_time_feature, cmap='viridis', aspect='auto')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Feature Index')
    ax4.set_title('Error Heatmap (Features vs Time)')
    plt.colorbar(im, ax=ax4)
    
    # 5. 样本间误差分布
    ax5 = fig.add_subplot(gs[1, :2])
    sample_total_errors = np.mean(np.abs(predictions - targets), axis=(1, 2))
    
    ax5.hist(sample_total_errors, bins=30, alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(sample_total_errors), color='red', linestyle='--', 
               label=f'Mean: {np.mean(sample_total_errors):.4f}')
    ax5.set_xlabel('Mean Absolute Error')
    ax5.set_ylabel('Number of Samples')
    ax5.set_title('Distribution of Sample Errors')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 特征重要性（基于预测误差）
    ax6 = fig.add_subplot(gs[1, 2:])
    feature_importance = 1 / (feature_errors + 1e-8)  # 误差越小，重要性越高
    feature_importance = feature_importance / np.sum(feature_importance)  # 归一化
    
    bars = ax6.bar(range(n_features), feature_importance, alpha=0.8)
    ax6.set_xlabel('Feature Index')
    ax6.set_ylabel('Relative Importance')
    ax6.set_title('Feature Importance (Based on Prediction Error)')
    ax6.grid(True, alpha=0.3)
    
    # 7. 相关性矩阵（特征间的预测相关性）
    ax7 = fig.add_subplot(gs[2, :2])
    pred_features_flat = predictions.reshape(-1, n_features)
    true_features_flat = targets.reshape(-1, n_features)
    
    corr_matrix = np.corrcoef(pred_features_flat.T, true_features_flat.T)[:n_features, n_features:]
    
    im = ax7.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax7.set_xlabel('True Feature Index')
    ax7.set_ylabel('Predicted Feature Index')
    ax7.set_title('Cross-Correlation Matrix (Pred vs True)')
    plt.colorbar(im, ax=ax7)
    
    # 8. 时序预测质量分析
    ax8 = fig.add_subplot(gs[2, 2:])
    # 计算每个时间步的平均相关性
    timestep_correlations = []
    for t in range(seq_len):
        pred_t = predictions[:, :, t].flatten()
        true_t = targets[:, :, t].flatten()
        if np.std(pred_t) > 1e-8 and np.std(true_t) > 1e-8:
            corr = np.corrcoef(pred_t, true_t)[0, 1]
            if not np.isnan(corr):
                timestep_correlations.append(corr)
            else:
                timestep_correlations.append(0)
        else:
            timestep_correlations.append(0)
    
    ax8.plot(range(len(timestep_correlations)), timestep_correlations, 'o-', linewidth=2)
    ax8.set_xlabel('Time Step')
    ax8.set_ylabel('Correlation')
    ax8.set_title('Prediction Quality over Time')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([-1, 1])
    
    plt.suptitle('Multi-dimensional Data Analysis', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'multidimensional_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate tfdiff model with enhanced visualizations')
    parser.add_argument('--task_id', type=int, default=4,
                       help='Task ID (0/1/2/3/4 for WiFi/FMCW/MIMO/EEG/Traffic)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-detect in model_dir)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions and targets to file')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip creating visualizations')
    parser.add_argument('--pred_len', type=int, default=20,
                       help='Prediction length for task 4 (traffic prediction)')
    # 🔧 新增参数
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['ConditionalUNet', 'tfdiff_WiFi', 'MaskedDiffusion'],
                       help='Force specific model type (default: auto-detect)')
    parser.add_argument('--skip_nan_check', action='store_true',
                       help='Skip NaN checking in predictions')
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(42)
    
    # 获取设备
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # 构建参数
    params = build_params(args.task_id, device.type, args.pred_len)
    
    # 重写批次大小
    params.batch_size = args.batch_size
    
    print(f"Evaluating task_id={args.task_id} with parameters:")
    print(f"  Model dir: {params.model_dir}")
    print(f"  Data dir: {params.data_dir}")
    print(f"  Batch size: {params.batch_size}")
    
    # 🔧 寻找checkpoint文件
    if args.checkpoint is None:
        # 尝试多个可能的检查点文件名
        possible_checkpoints = [
            os.path.join(params.model_dir, 'best_model.pt'),
            os.path.join(params.model_dir, 'traffic_model_final.pt'),
            os.path.join(params.model_dir, 'checkpoint.pt'),
            os.path.join(params.model_dir, 'weights.pt'),
            os.path.join(params.model_dir, 'model.pt'),
            os.path.join(params.model_dir, 'best_masked_diffusion_model.pt'),  # 🔧 新增
        ]
        
        checkpoint_path = None
        for path in possible_checkpoints:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in {params.model_dir}. "
                                   f"Tried: {possible_checkpoints}")
    else:
        checkpoint_path = args.checkpoint
    
    print(f"📁 Using checkpoint: {checkpoint_path}")
    
    # 🔧 检测并创建适当的模型
    model_type = args.model_type or detect_model_type(checkpoint_path)
    
    if model_type == 'MaskedDiffusion' and MASKED_DIFFUSION_AVAILABLE:
        # 创建修复版扩散模型
        from wifi_model import MaskedDiffusionConfig
        config = MaskedDiffusionConfig()
        config.device = device.type
        config.input_dim = params.input_dim
        config.cond_dim = getattr(params, 'cond_dim', 148)
        config.seq_len = getattr(params, 'seq_len', 10)
        config.pred_len = getattr(params, 'pred_len', 10)
        
        model = create_masked_diffusion_model(config)
        diffusion = None  # 修复版模型内置了扩散过程
        print("🏗️ Created MaskedDiffusion model")
        
    elif model_type == 'ConditionalUNet':
        from model import ConditionalUNet
        model = ConditionalUNet(params).to(device)
        diffusion = create_diffusion(params, device)
        print("🏗️ Created ConditionalUNet model")
        
    else:
        model = tfdiff_WiFi(params).to(device)
        diffusion = create_diffusion(params, device)
        print("🏗️ Created tfdiff_WiFi model")
    
    # 创建数据加载器（评估模式）
    print("Loading dataset...")
    dataset = from_path(params)
    
    # 验证数据集
    try:
        first_batch = next(iter(dataset))
        print(f"✅ Dataset loaded successfully")
        if isinstance(first_batch, dict):
            print(f"  Batch keys: {list(first_batch.keys())}")
            for key, value in first_batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape}")
        else:
            print(f"  Batch shape: {first_batch.shape}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")
    
    # 加载检查点
    load_model_checkpoint(model, checkpoint_path, device)
    
    # 评估模型
    print("Starting model evaluation...")
    metrics, predictions, targets = evaluate_model(
        model, diffusion, dataset, device, params, args.num_samples
    )
    
    # 🔧 检查结果的有效性
    if not args.skip_nan_check:
        pred_nan_count = np.isnan(predictions.cpu().numpy()).sum() if torch.is_tensor(predictions) else np.isnan(predictions).sum()
        target_nan_count = np.isnan(targets.cpu().numpy()).sum() if torch.is_tensor(targets) else np.isnan(targets).sum()
        
        if pred_nan_count > 0:
            print(f"⚠️ Found {pred_nan_count} NaN values in predictions")
        if target_nan_count > 0:
            print(f"⚠️ Found {target_nan_count} NaN values in targets")
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:15}: {metric_value:.6f}")
    print("="*60)
    
    # 创建增强的可视化
    if not args.no_visualization:
        print("\n🎨 Creating comprehensive visualizations...")
        create_comprehensive_visualizations(predictions, targets, metrics, args.output_dir, args.task_id)
        print("✅ All visualizations completed!")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics_path = os.path.join(args.output_dir, f'metrics_task{args.task_id}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📝 Metrics saved to: {metrics_path}")
    
    # 保存预测结果（如果requested）
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, f'predictions_task{args.task_id}.npz')
        np.savez(predictions_path, 
                predictions=predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions, 
                targets=targets.cpu().numpy() if torch.is_tensor(targets) else targets)
        print(f"💾 Predictions saved to: {predictions_path}")
    
    print("\n🎉 Evaluation completed successfully!")
    print(f"📁 All results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
