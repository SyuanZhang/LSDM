import os
from pathlib import Path
import torch
from torch.cuda import device_count
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel

from argparse import ArgumentParser

from params import all_params
from learner import tfdiffLearner
# 修改导入，使用支持mask的模型
try:
    from wifi_model import tfdiff_WiFi, MaskedDiffusionTrainer, MaskedDiffusionLoss, MaskedDiffusionConfig
    MASK_MODEL_AVAILABLE = True
except ImportError:
    from wifi_model import tfdiff_WiFi
    MASK_MODEL_AVAILABLE = False
    print("⚠️ Masked model not available, using original model")

from dataset import from_path
from model import ConditionalUNet

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

def _train_impl(replica_id, model, dataset, params):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=1e-6, eps=1e-8, betas=(0.9, 0.999)
    )
    learner = tfdiffLearner(params.log_dir, params.model_dir, model, dataset, opt, params)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()

    for g in opt.param_groups:
        g['lr'] = params.learning_rate

    learner.train(max_iter=params.max_iter)

def safe_save_model(model_state, save_path, description="model"):
    """安全保存模型的函数"""
    try:
        # 确保目录存在
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用临时文件名避免冲突
        temp_path = str(save_path) + '.tmp'
        
        # 保存到临时文件
        torch.save(model_state, temp_path)
        
        # 原子性替换
        if Path(save_path).exists():
            Path(save_path).unlink()  # 删除旧文件
        Path(temp_path).rename(save_path)  # 重命名临时文件
        
        print(f"🏆 {description} saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to save {description}: {e}")
        # 清理临时文件
        temp_path = str(save_path) + '.tmp'
        if Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except:
                pass
        return False

def verify_dataset_split(dataset):
    """正确验证数据集用户划分"""
    print("🔍 验证数据集划分方式...")
    
    all_users = set()
    batch_count = 0
    total_samples = 0
    
    try:
        for batch_idx, batch in enumerate(dataset):
            if isinstance(batch, dict):
                # 处理不同的用户ID字段名
                user_ids = None
                if 'user_ids' in batch:
                    user_ids = batch['user_ids']
                elif 'user_id' in batch:
                    user_ids = batch['user_id']
                    if not isinstance(user_ids, (list, tuple)):
                        user_ids = [user_ids]
                
                if user_ids is not None:
                    if isinstance(user_ids, torch.Tensor):
                        user_ids = user_ids.tolist()
                    elif not isinstance(user_ids, (list, tuple)):
                        user_ids = [user_ids]
                    
                    all_users.update(user_ids)
                    total_samples += len(user_ids)
                    
                    # 只显示前3个批次的详情
                    if batch_idx < 3:
                        print(f"  • Batch {batch_idx}: {len(user_ids)} users = {user_ids}")
                    elif batch_idx == 3:
                        print(f"  • ... (继续处理剩余批次)")
                
                batch_count += 1
                
                # 防止数据集太大时的无限循环
                if batch_count >= 100:  
                    print(f"  • 已检查 {batch_count} 个批次，停止验证以节省时间")
                    break
        
        print(f"✅ 数据集验证完成:")
        print(f"  • 检查批次数: {batch_count}")
        print(f"  • 总样本数: {total_samples}")
        print(f"  • 唯一用户数: {len(all_users)}")
        if all_users:
            print(f"  • 用户ID范围: [{min(all_users)}, {max(all_users)}]")
            # 显示一些用户ID样本
            user_sample = sorted(list(all_users))[:20]
            print(f"  • 用户ID示例: {user_sample}{'...' if len(all_users) > 20 else ''}")
        
        return all_users, total_samples, batch_count
        
    except Exception as e:
        print(f"⚠️ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return set(), 0, 0

def analyze_data_format(sample_batch, params):
    """详细分析数据格式并返回正确的维度信息"""
    print("🔍 详细分析数据格式...")
    
    if not isinstance(sample_batch, dict) or 'data' not in sample_batch:
        print("❌ 无效的批次格式")
        return None
    
    data = sample_batch['data']
    print(f"📊 数据形状分析: {data.shape}")
    
    batch_size = data.shape[0]
    
    # 获取参数配置
    pred_len = getattr(params, 'pred_len', 20)
    total_seq_len = getattr(params, 'total_seq_len', 168)
    input_dim = getattr(params, 'input_dim', 20)
    
    format_info = {
        'batch_size': batch_size,
        'original_shape': data.shape,
        'time_dim_index': None,
        'time_length': None,
        'feature_dim_index': None,
        'feature_length': None,
        'format_type': None
    }
    
    if data.dim() == 2:  # [B, F]
        print("  • 格式: [Batch, Features]")
        format_info['format_type'] = '2D'
        format_info['feature_dim_index'] = 1
        format_info['feature_length'] = data.shape[1]
        print("  ⚠️ 2D数据无法应用时序mask")
        
    elif data.dim() == 3:  # [B, T, F] 
        print("  • 格式: [Batch, Time, Features]")
        format_info['format_type'] = '3D'
        format_info['time_dim_index'] = 1
        format_info['time_length'] = data.shape[1]
        format_info['feature_dim_index'] = 2
        format_info['feature_length'] = data.shape[2]
        
    elif data.dim() == 4:  # [B, T, F, 1] 或 [B, C, T, F]
        dim1, dim2, dim3 = data.shape[1], data.shape[2], data.shape[3]
        
        # 判断格式
        if dim3 == 1:  # [B, T, F, 1] 格式
            print(f"  • 格式: [Batch={batch_size}, Time={dim1}, Features={dim2}, Channel={dim3}]")
            format_info['format_type'] = '4D_BTFC'
            format_info['time_dim_index'] = 1
            format_info['time_length'] = dim1
            format_info['feature_dim_index'] = 2
            format_info['feature_length'] = dim2
            
        elif dim1 == 1:  # [B, 1, T, F] 格式
            print(f"  • 格式: [Batch={batch_size}, Channel={dim1}, Time={dim2}, Features={dim3}]")
            format_info['format_type'] = '4D_BCTF'
            format_info['time_dim_index'] = 2
            format_info['time_length'] = dim2
            format_info['feature_dim_index'] = 3
            format_info['feature_length'] = dim3
            
        else:  # 通用 [B, C, T, F] 格式，需要启发式判断
            # 根据配置参数推断
            if dim1 == pred_len and dim2 == input_dim:
                print(f"  • 推断格式: [Batch={batch_size}, Time={dim1}, Features={dim2}, Extra={dim3}]")
                format_info['format_type'] = '4D_BTFC'
                format_info['time_dim_index'] = 1
                format_info['time_length'] = dim1
                format_info['feature_dim_index'] = 2
                format_info['feature_length'] = dim2
            elif dim2 == pred_len and dim3 == input_dim:
                print(f"  • 推断格式: [Batch={batch_size}, Channel={dim1}, Time={dim2}, Features={dim3}]")
                format_info['format_type'] = '4D_BCTF'
                format_info['time_dim_index'] = 2
                format_info['time_length'] = dim2
                format_info['feature_dim_index'] = 3
                format_info['feature_length'] = dim3
            else:
                print(f"  ⚠️ 无法确定维度含义，使用默认: Time在索引2")
                format_info['format_type'] = '4D_BCTF'
                format_info['time_dim_index'] = 2
                format_info['time_length'] = dim2
                format_info['feature_dim_index'] = 3
                format_info['feature_length'] = dim3
                
    elif data.dim() == 5:  # [B, C, T, F, E]
        print(f"  • 格式: [Batch={batch_size}, C={data.shape[1]}, Time={data.shape[2]}, F={data.shape[3]}, E={data.shape[4]}]")
        format_info['format_type'] = '5D'
        format_info['time_dim_index'] = 2
        format_info['time_length'] = data.shape[2]
        format_info['feature_dim_index'] = 3
        format_info['feature_length'] = data.shape[3]
        
    else:
        print(f"  ❌ 不支持的维度: {data.dim()}")
        return None
    
    # 验证时间长度
    if format_info['time_length'] is not None:
        print(f"  ✅ 检测到时间维度: 索引{format_info['time_dim_index']}, 长度{format_info['time_length']}")
        mask_length = getattr(params, 'mask_length', pred_len)
        if mask_length >= format_info['time_length']:
            print(f"  ⚠️ Mask长度({mask_length}) >= 时间长度({format_info['time_length']})")
        else:
            print(f"  ✅ Mask长度({mask_length})配置正常")
    
    return format_info

def create_smart_mask(batch_size, format_info, mask_length, device, strategy='prefix'):
    """根据数据格式智能创建mask"""
    if format_info is None or format_info['time_length'] is None:
        print("❌ 无法创建mask：时间维度未知")
        return None
    
    seq_len = format_info['time_length']
    time_dim = format_info['time_dim_index']
    
    # 调整mask长度
    adjusted_mask_length = min(mask_length, seq_len - 1)
    if adjusted_mask_length != mask_length:
        print(f"🔧 Mask长度调整: {mask_length} -> {adjusted_mask_length}")
    
    # 创建基础mask [B, T]
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    if strategy == 'prefix':
        mask[:, :adjusted_mask_length] = True
    elif strategy == 'suffix':
        mask[:, -adjusted_mask_length:] = True
    elif strategy == 'random':
        for b in range(batch_size):
            indices = torch.randperm(seq_len)[:adjusted_mask_length]
            mask[b, indices] = True
    
    return mask, adjusted_mask_length

def apply_smart_mask(data, mask, format_info, mask_value=0.0):
    """根据数据格式智能应用mask"""
    if format_info is None or mask is None:
        return data
    
    masked_data = data.clone()
    time_dim = format_info['time_dim_index']
    
    try:
        if format_info['format_type'] == '2D':
            print("⚠️ 2D数据跳过mask应用")
            return masked_data
            
        elif format_info['format_type'] == '3D':  # [B, T, F]
            # mask: [B, T] -> [B, T, F]
            mask_expanded = mask.unsqueeze(-1).expand_as(data)
            masked_data[mask_expanded] = mask_value
            
        elif format_info['format_type'] == '4D_BTFC':  # [B, T, F, 1]
            # mask: [B, T] -> [B, T, F, 1]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(data)
            masked_data[mask_expanded] = mask_value
            
        elif format_info['format_type'] == '4D_BCTF':  # [B, C, T, F]
            # mask: [B, T] -> [B, C, T, F]
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand_as(data)
            masked_data[mask_expanded] = mask_value
            
        elif format_info['format_type'] == '5D':  # [B, C, T, F, E]
            # mask: [B, T] -> [B, C, T, F, E]
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(data)
            masked_data[mask_expanded] = mask_value
            
        else:
            print(f"⚠️ 未知格式类型: {format_info['format_type']}")
            return masked_data
            
    except Exception as e:
        print(f"❌ Mask应用失败: {e}")
        return masked_data
    
    return masked_data

def compute_smart_masked_loss(pred, target, mask, format_info, loss_fn):
    """根据数据格式智能计算mask损失"""
    if format_info is None or mask is None:
        return loss_fn(pred, target)
    
    try:
        # 根据格式扩展mask
        if format_info['format_type'] == '3D':  # [B, T, F]
            mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        elif format_info['format_type'] == '4D_BTFC':  # [B, T, F, 1]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred)
        elif format_info['format_type'] == '4D_BCTF':  # [B, C, T, F]
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand_as(pred)
        elif format_info['format_type'] == '5D':  # [B, C, T, F, E]
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(pred)
        else:
            return loss_fn(pred, target)
        
        # 计算mask区域的损失
        if mask_expanded.sum() > 0:
            masked_pred = pred[mask_expanded]
            masked_target = target[mask_expanded]
            return loss_fn(masked_pred, masked_target)
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
    except Exception as e:
        print(f"⚠️ Mask损失计算失败: {e}")
        return loss_fn(pred, target)

def train_with_mask(params):
    """支持mask的训练函数"""
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Using device: {device}")
    print(f"🎭 Mask-aware training enabled!")
    
    # 确保目录存在
    model_dir = Path(params.model_dir)
    log_dir = Path(params.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义checkpoint目录和文件路径
    checkpoint_dir = model_dir
    best_model_path = checkpoint_dir / 'best_masked_model.pt'
    final_model_path = checkpoint_dir / 'masked_traffic_model_final.pt'
    
    print(f"📁 Directories:")
    print(f"  • Model dir: {model_dir}")
    print(f"  • Log dir: {log_dir}")
    
    # 创建数据集
    print("📊 Creating dataset...")
    dataset = from_path(params)
    
    # 验证数据集
    all_users, total_samples, batch_count = verify_dataset_split(dataset)
    
    if not all_users:
        print("❌ 警告: 无法获取用户信息，可能使用了错误的数据集格式")
        print("   将继续训练但无法验证用户划分...")
    
    # 分析数据格式
    print("🔍 分析数据格式...")
    try:
        sample_batch = next(iter(dataset))
        format_info = analyze_data_format(sample_batch, params)
        if format_info is None:
            print("❌ 数据格式分析失败，回退到原始训练")
            return train_original(params)
    except Exception as e:
        print(f"❌ 无法获取样本数据: {e}")
        return train_original(params)
    
    # 创建支持mask的模型
    print("🏗️ Creating masked diffusion model...")
    
    # 如果有可用的mask模型，使用专门的训练器
    if MASK_MODEL_AVAILABLE:
        # 创建配置
        config = MaskedDiffusionConfig()
        
        # 从params更新配置
        config.input_dim = getattr(params, 'input_dim', 20)
        config.hidden_dim = getattr(params, 'hidden_dim', 256)
        config.cond_dim = getattr(params, 'cond_dim', 148)
        config.device = device
        config.batch_size = params.batch_size
        config.learning_rate = getattr(params, 'learning_rate', 1e-4)
        config.seq_len = format_info['time_length']
        
        # Mask相关配置
        config.mask_length = getattr(params, 'mask_length', params.pred_len)
        config.mask_strategies = getattr(params, 'mask_strategies', ['prefix'])
        config.mask_weight = getattr(params, 'mask_weight', 1.0)
        config.unmask_weight = getattr(params, 'unmask_weight', 0.1)
        
        print(f"🎭 Mask Configuration:")
        print(f"  • Mask length: {config.mask_length}")
        print(f"  • Mask strategies: {config.mask_strategies}")
        print(f"  • Mask weight: {config.mask_weight}")
        print(f"  • Unmask weight: {config.unmask_weight}")
        
        # 创建模型和训练器
        net = tfdiff_WiFi(config).to(device)
        trainer = MaskedDiffusionTrainer(net, config)
        
        print(f"📊 Model parameters: {sum(p.numel() for p in net.parameters()):,}")
        
        # 使用专门的训练器训练
        return train_with_masked_trainer(trainer, dataset, config, checkpoint_dir, params.max_iter)
    
    else:
        # 回退到原始训练方式，但集成mask功能
        net = ConditionalUNet(params).to(device)
        print(f"📊 Model parameters: {sum(p.numel() for p in net.parameters()):,}")
        
        # 使用修改后的训练逻辑
        return train_with_manual_mask(net, dataset, params, device, checkpoint_dir, format_info)

def train_with_masked_trainer(trainer, dataset, config, checkpoint_dir, max_iter):
    """使用专门的mask训练器进行训练"""
    print("🚀 Starting mask-aware training with specialized trainer...")
    
    best_loss = float('inf')
    iteration = 0
    loss_history = []
    
    try:
        data_loader = iter(dataset)
    except Exception as e:
        print(f"❌ Error creating data loader: {e}")
        return
    
    while iteration < max_iter:
        for batch_idx, batch in enumerate(data_loader):
            if iteration >= max_iter:
                print(f"🏁 Reached maximum iterations ({max_iter})")
                break
            
            try:
                # 使用专门的训练器进行训练步骤
                total_loss, mask_loss, unmask_loss = trainer.train_step(batch)
                
                if total_loss == float('inf'):
                    continue
                
                loss_history.append(total_loss)
                
                # 更新最佳损失
                if total_loss < best_loss:
                    best_loss = total_loss
                    model_state = {
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'scheduler_state_dict': trainer.scheduler.state_dict(),
                        'total_loss': total_loss,
                        'mask_loss': mask_loss,
                        'unmask_loss': unmask_loss,
                        'iteration': iteration,
                        'best_loss': best_loss,
                    }
                    safe_save_model(model_state, checkpoint_dir / 'best_masked_model.pt', "Best masked model")
                
                # 打印训练进度
                if iteration % 1000 == 0:
                    current_lr = trainer.optimizer.param_groups[0]['lr']
                    print(f"Iter {iteration:4d}/{max_iter} | "
                          f"Total: {total_loss:.6f} | "
                          f"Mask: {mask_loss:.6f} | "
                          f"Unmask: {unmask_loss:.6f} | "
                          f"Best: {best_loss:.6f} | "
                          f"LR: {current_lr:.2e}")
                
                # 定期保存checkpoint
                if iteration > 0 and iteration % 5000 == 0:
                    checkpoint_path = checkpoint_dir / f"masked_checkpoint_iter_{iteration}.pt"
                    checkpoint_state = {
                        'iteration': iteration,
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'scheduler_state_dict': trainer.scheduler.state_dict(),
                        'total_loss': total_loss,
                        'mask_loss': mask_loss,
                        'unmask_loss': unmask_loss,
                        'loss_history': loss_history,
                        'best_loss': best_loss,
                    }
                    safe_save_model(checkpoint_state, checkpoint_path, f"Masked checkpoint {iteration}")
                
                iteration += 1
                
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 重新创建数据加载器
        try:
            data_loader = iter(dataset)
        except Exception as e:
            print(f"❌ Error recreating data loader: {e}")
            break
    
    # 保存最终模型
    final_model_state = {
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'loss_history': loss_history,
        'best_loss': best_loss,
        'final_loss': loss_history[-1] if loss_history else float('inf'),
        'total_iterations': iteration,
    }
    
    safe_save_model(final_model_state, checkpoint_dir / 'masked_traffic_model_final.pt', "Final masked model")
    
    print(f"\n🏁 Masked training completed!")
    print(f"  • Total iterations: {iteration}")
    print(f"  • Best loss: {best_loss:.6f}")
    
    if len(loss_history) > 10:
        try:
            plot_training_loss(loss_history, checkpoint_dir, prefix="masked_")
        except Exception as e:
            print(f"⚠️ Could not save loss plot: {e}")

def train_with_manual_mask(net, dataset, params, device, checkpoint_dir, format_info):
    """使用手动mask实现的训练（完全修复版本）"""
    print("🚀 Starting training with manual mask implementation...")
    
    # 优化器设置
    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr=params.learning_rate, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=params.max_iter, 
        eta_min=1e-7
    )
    
    # 损失函数
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    
    def combined_loss(pred, target):
        mse = mse_loss(pred, target)
        l1 = l1_loss(pred, target)
        return mse + 0.1 * l1
    
    # 训练变量
    net.train()
    iteration = 0
    best_loss = float('inf')
    loss_history = []
    mask_length = getattr(params, 'mask_length', params.pred_len)
    mask_strategy = getattr(params, 'mask_strategy', 'prefix')
    
    # 早期停止参数
    patience = getattr(params, 'early_stopping_patience', 200)
    no_improve_count = 0
    best_loss_iteration = 0
    
    print(f"📋 Training parameters:")
    print(f"  • Learning rate: {params.learning_rate}")
    print(f"  • Max iterations: {params.max_iter}")
    print(f"  • Batch size: {params.batch_size}")
    print(f"  • Mask length: {mask_length}")
    print(f"  • Mask strategy: {mask_strategy}")
    print(f"  • Data format: {format_info['format_type']}")
    print(f"  • Time dimension: index {format_info['time_dim_index']}, length {format_info['time_length']}")
    print(f"  • Early stopping patience: {patience}")
    
    # 训练循环
    while iteration < params.max_iter:
        try:
            data_loader = iter(dataset)
        except Exception as e:
            print(f"❌ Error creating data loader: {e}")
            break
        
        for batch_idx, batch in enumerate(data_loader):
            if iteration >= params.max_iter:
                break
            
            try:
                # 获取数据
                if isinstance(batch, dict):
                    data = batch['data'].to(device)
                    cond = batch['cond'].to(device) if 'cond' in batch else None
                else:
                    data = batch[0].to(device)
                    cond = batch[1].to(device) if len(batch) > 1 else None
                
                # 调试信息：第一次迭代时打印详细信息
                if iteration == 0:
                    print(f"🔍 Debug info - Batch {batch_idx}:")
                    print(f"  • Data shape: {data.shape}")
                    print(f"  • Data type: {data.dtype}")
                    print(f"  • Data range: [{data.min().item():.4f}, {data.max().item():.4f}]")
                    if cond is not None:
                        print(f"  • Cond shape: {cond.shape}")
                        print(f"  • Cond range: [{cond.min().item():.4f}, {cond.max().item():.4f}]")
                
                # 验证数据形状与分析结果一致
                if data.shape != format_info['original_shape']:
                    print(f"⚠️ 数据形状不一致! 期望: {format_info['original_shape']}, 实际: {data.shape}")
                    # 重新分析当前批次
                    temp_format = analyze_data_format({'data': data}, params)
                    if temp_format:
                        format_info = temp_format
                
                batch_size = data.shape[0]
                
                # 创建智能mask
                mask, adjusted_mask_length = create_smart_mask(
                    batch_size, format_info, mask_length, device, mask_strategy
                )
                
                if mask is None:
                    print(f"⚠️ 无法创建mask，跳过批次 {batch_idx}")
                    continue
                
                # 验证mask创建
                if iteration == 0:
                    print(f"  • Mask shape: {mask.shape}")
                    print(f"  • Adjusted mask length: {adjusted_mask_length}")
                    print(f"  • Mask ratio: {mask.float().mean().item():.3f}")
                
                # 原始数据（目标）
                target_data = data.clone()
                
                # 应用智能mask
                masked_data = apply_smart_mask(data, mask, format_info)
                
                # 验证mask应用
                if iteration == 0:
                    mask_applied = not torch.equal(data, masked_data)
                    print(f"  • Mask applied successfully: {mask_applied}")
                    if mask_applied:
                        diff_count = (data != masked_data).sum().item()
                        total_elements = data.numel()
                        diff_ratio = diff_count / total_elements
                        print(f"  • Masked elements: {diff_count}/{total_elements} ({diff_ratio:.3f})")
                
                # 扩散模型训练步骤
                t = torch.randint(0, 1000, (batch_size,), device=device)
                
                # 噪声调度
                beta_start, beta_end = 1e-4, 0.02
                betas = torch.linspace(beta_start, beta_end, 1000, device=device)
                alphas = 1. - betas
                alphas_cumprod = torch.cumprod(alphas, dim=0)
                
                # 生成噪声并添加到mask数据
                noise = torch.randn_like(masked_data)
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t])
                
                # 重塑系数以匹配数据维度
                shape = [-1] + [1] * (masked_data.dim() - 1)
                sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(shape)
                sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(shape)
                
                # 添加噪声
                noisy_data = sqrt_alphas_cumprod * masked_data + sqrt_one_minus_alphas_cumprod * noise
                
                # 前向传播
                optimizer.zero_grad()
                predicted = net(noisy_data, t, cond)
                
                # 智能计算损失
                mask_loss = compute_smart_masked_loss(predicted, target_data, mask, format_info, combined_loss)
                global_loss = combined_loss(predicted, target_data) * 0.1
                total_loss = mask_loss + global_loss
                
                # L2正则化
                l2_reg = 0
                for param in net.parameters():
                    l2_reg += torch.norm(param, 2)
                total_loss += 1e-5 * l2_reg
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # 记录损失
                current_loss = total_loss.item()
                loss_history.append(current_loss)
                
                # 早期停止逻辑
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_loss_iteration = iteration
                    no_improve_count = 0
                    
                    # 保存最佳模型
                    model_state = {
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': current_loss,
                        'iteration': iteration,
                        'mask_length': mask_length,
                        'mask_strategy': mask_strategy,
                        'format_info': format_info,
                    }
                    safe_save_model(model_state, checkpoint_dir / 'best_manual_masked_model.pt', "Best manual masked model")
                else:
                    no_improve_count += 1
                
                # 检查早期停止
                if no_improve_count >= patience:
                    print(f"🛑 Early stopping at iteration {iteration}")
                    print(f"   No improvement for {patience} iterations")
                    print(f"   Best loss: {best_loss:.6f} at iteration {best_loss_iteration}")
                    break
                
                # 打印进度
                if iteration % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Iter {iteration:4d}/{params.max_iter} | "
                          f"Loss: {current_loss:.6f} | "
                          f"Best: {best_loss:.6f} | "
                          f"NoImprove: {no_improve_count}/{patience} | "
                          f"LR: {current_lr:.2e}")
                
                # 详细监控
                if iteration % 50 == 0 and iteration > 0:
                    # 计算梯度范数
                    total_norm = 0
                    for p in net.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    
                    # 计算学习进度
                    if len(loss_history) >= 100:
                        recent_avg = sum(loss_history[-50:]) / 50
                        early_avg = sum(loss_history[:50]) / 50
                        improvement = (early_avg - recent_avg) / early_avg * 100
                        
                        print(f"  📈 Grad norm: {total_norm:.4f} | "
                              f"Improvement: {improvement:.1f}% | "
                              f"Recent avg: {recent_avg:.6f}")
                
                # 定期保存checkpoint
                if iteration > 0 and iteration % 1000 == 0:
                    checkpoint_path = checkpoint_dir / f"manual_masked_checkpoint_iter_{iteration}.pt"
                    checkpoint_state = {
                        'iteration': iteration,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': current_loss,
                        'loss_history': loss_history,
                        'best_loss': best_loss,
                        'mask_length': mask_length,
                        'mask_strategy': mask_strategy,
                        'format_info': format_info,
                        'no_improve_count': no_improve_count,
                    }
                    safe_save_model(checkpoint_state, checkpoint_path, f"Manual masked checkpoint {iteration}")
                
                iteration += 1
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 如果早期停止，跳出外层循环
        if no_improve_count >= patience:
            break
    
    # 保存最终模型
    final_model_state = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
        'best_loss': best_loss,
        'final_loss': loss_history[-1] if loss_history else float('inf'),
        'total_iterations': iteration,
        'mask_length': mask_length,
        'mask_strategy': mask_strategy,
        'format_info': format_info,
        'early_stopped': no_improve_count >= patience,
    }
    
    safe_save_model(final_model_state, checkpoint_dir / 'manual_masked_traffic_model_final.pt', "Final manual masked model")
    
    print(f"\n🏁 Manual masked training completed!")
    print(f"  • Total iterations: {iteration}")
    print(f"  • Best loss: {best_loss:.6f} (at iteration {best_loss_iteration})")
    print(f"  • Final loss: {loss_history[-1] if loss_history else 'N/A'}")
    print(f"  • Early stopped: {no_improve_count >= patience}")
    
    # 绘制训练曲线
    if len(loss_history) > 10:
        try:
            plot_training_loss(loss_history, checkpoint_dir, prefix="manual_masked_")
        except Exception as e:
            print(f"⚠️ Could not save loss plot: {e}")

def train(params):
    """主训练函数，自动选择训练方式"""
    # 检查是否启用mask模式
    use_mask = getattr(params, 'use_mask', False) or getattr(params, 'prediction_mode', False)
    
    if use_mask and MASK_MODEL_AVAILABLE:
        print("🎭 Using mask-aware training mode with specialized trainer")
        train_with_mask(params)
    elif use_mask:
        print("🎭 Using manual mask implementation")
        train_with_mask(params)
    else:
        print("📊 Using standard training mode")
        train_original(params)

def train_original(params):
    """原始训练函数（保持不变但添加了改进）"""
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Using device: {device}")
    
    # 确保目录存在
    model_dir = Path(params.model_dir)
    log_dir = Path(params.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义checkpoint目录和文件路径
    checkpoint_dir = model_dir
    best_model_path = checkpoint_dir / 'best_model.pt'
    final_model_path = checkpoint_dir / 'traffic_model_final.pt'
    
    print(f"📁 Directories:")
    print(f"  • Model dir: {model_dir}")
    print(f"  • Log dir: {log_dir}")
    
    # 创建数据集
    print("📊 Creating dataset...")
    dataset = from_path(params)
    
    # 验证数据集
    all_users, total_samples, batch_count = verify_dataset_split(dataset)
    
    if not all_users:
        print("❌ 警告: 无法获取用户信息，可能使用了错误的数据集格式")
        print("   将继续训练但无法验证用户划分...")
    
    # 创建模型
    print("🏗️ Creating model...")
    net = ConditionalUNet(params).to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # 优化器设置
    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr=params.learning_rate, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=params.max_iter, 
        eta_min=1e-7
    )
    
    # 损失函数
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    
    def combined_loss(pred, target):
        mse = mse_loss(pred, target)
        l1 = l1_loss(pred, target)
        return mse + 0.1 * l1
    
    # 训练变量
    net.train()
    epoch = 0
    iteration = 0
    running_loss = 0.0
    best_loss = float('inf')
    loss_history = []
    
    # 早期停止
    patience = getattr(params, 'early_stopping_patience', 500)
    no_improve_count = 0
    
    print(f"🚀 Starting training with enhanced settings...")
    print(f"📋 Training parameters:")
    print(f"  • Learning rate: {params.learning_rate}")
    print(f"  • Max iterations: {params.max_iter}")
    print(f"  • Batch size: {params.batch_size}")
    print(f"  • Dataset batches: {batch_count}")
    print(f"  • Total samples: {total_samples}")
    print(f"  • Early stopping patience: {patience}")
    
    # 训练循环
    while iteration < params.max_iter:
        epoch += 1
        print(f"\n📅 Epoch {epoch} starting...")
        
        # 重新创建数据加载器确保每个epoch的随机性
        try:
            data_loader = iter(dataset)
        except Exception as e:
            print(f"❌ Error creating data loader: {e}")
            break
        
        epoch_loss = 0.0
        epoch_batches = 0
        
        # 遍历当前epoch的所有批次
        for batch_idx, batch in enumerate(data_loader):
            if iteration >= params.max_iter:
                print(f"🏁 Reached maximum iterations ({params.max_iter})")
                break
            
            try:
                # 验证批次格式
                if not isinstance(batch, dict):
                    print(f"⚠️ Batch {batch_idx}: Not a dictionary, got {type(batch)}")
                    continue
                
                if 'data' not in batch:
                    print(f"⚠️ Batch {batch_idx}: Missing 'data' key, keys: {list(batch.keys())}")
                    continue
                
                # 获取数据
                data = batch['data'].to(device)
                cond = batch['cond'].to(device) if 'cond' in batch else None
                
                # 验证数据形状
                if data.dim() != 4:
                    if data.dim() == 3:  # [B, T, F] -> [B, 1, T, F]
                        data = data.unsqueeze(1)
                    elif data.dim() == 2:  # [B, F] -> [B, 1, 1, F]
                        data = data.unsqueeze(1).unsqueeze(1)
                    else:
                        print(f"⚠️ Unexpected data shape: {data.shape}")
                        continue
                
                # 首次迭代时显示数据统计
                if iteration == 0:
                    print(f"\n📊 First batch statistics:")
                    print(f"  • Data shape: {data.shape}")
                    print(f"  • Data range: [{data.min().item():.4f}, {data.max().item():.4f}]")
                    print(f"  • Data mean: {data.mean().item():.4f}, std: {data.std().item():.4f}")
                    if cond is not None:
                        print(f"  • Cond shape: {cond.shape}")
                        print(f"  • Cond range: [{cond.min().item():.4f}, {cond.max().item():.4f}]")
                    print(f"  • Batch keys: {list(batch.keys())}")
                
                # 扩散模型训练步骤
                batch_size = data.shape[0]
                
                # 随机时间步
                t = torch.randint(0, 1000, (batch_size,), device=device)
                
                # 噪声调度参数
                beta_start, beta_end = 1e-4, 0.02
                betas = torch.linspace(beta_start, beta_end, 1000, device=device)
                alphas = 1. - betas
                alphas_cumprod = torch.cumprod(alphas, dim=0)
                
                # 生成噪声
                noise = torch.randn_like(data)
                
                # 计算噪声系数
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t])
                
                # 重塑系数以匹配数据维度
                shape = [-1] + [1] * (data.dim() - 1)
                sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(shape)
                sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(shape)
                
                # 添加噪声到数据
                noisy_data = sqrt_alphas_cumprod * data + sqrt_one_minus_alphas_cumprod * noise
                
                # 前向传播
                optimizer.zero_grad()
                
                predicted_noise = net(noisy_data, t, cond)
                
                # 计算损失
                loss = combined_loss(predicted_noise, noise)
                
                # L2正则化
                l2_reg = 0
                for param in net.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += 1e-5 * l2_reg
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                
                optimizer.step()
                scheduler.step()
                
                # 记录损失
                current_loss = loss.item()
                running_loss += current_loss
                epoch_loss += current_loss
                loss_history.append(current_loss)
                epoch_batches += 1
                
                # 早期停止逻辑
                if current_loss < best_loss:
                    best_loss = current_loss
                    no_improve_count = 0
                    model_state = {
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': current_loss,
                        'iteration': iteration,
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                    }
                    safe_save_model(model_state, best_model_path, "Best model")
                else:
                    no_improve_count += 1
                
                # 检查早期停止
                if no_improve_count >= patience:
                    print(f"🛑 Early stopping triggered at iteration {iteration}")
                    break
                
                # 打印训练进度
                if iteration % 10 == 0:
                    avg_loss = running_loss / (iteration + 1)
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    print(f"Iter {iteration:4d}/{params.max_iter} | "
                          f"Epoch {epoch:2d} Batch {batch_idx:2d} | "
                          f"Loss: {current_loss:.6f} | "
                          f"Best: {best_loss:.6f} | "
                          f"Avg: {avg_loss:.6f} | "
                          f"NoImprove: {no_improve_count}/{patience} | "
                          f"LR: {current_lr:.2e}")
                
                # 详细监控
                if iteration % 50 == 0 and iteration > 0:
                    # 计算梯度范数
                    total_norm = 0
                    for p in net.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    
                    pred_range = f"[{predicted_noise.min().item():.4f}, {predicted_noise.max().item():.4f}]"
                    noise_range = f"[{noise.min().item():.4f}, {noise.max().item():.4f}]"
                    
                    print(f"  📈 Grad norm: {total_norm:.6f} | Pred: {pred_range} | True: {noise_range}")
                    
                    # 学习进度分析
                    if iteration >= 100:
                        recent_avg = sum(loss_history[-50:]) / min(50, len(loss_history))
                        early_avg = sum(loss_history[:50]) / min(50, len(loss_history))
                        improvement = (early_avg - recent_avg) / early_avg * 100
                        print(f"  📊 Improvement: {improvement:.2f}% (from {early_avg:.4f} to {recent_avg:.4f})")
                
                # 定期保存checkpoint
                if iteration > 0 and iteration % 5000 == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
                    checkpoint_state = {
                        'iteration': iteration,
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': current_loss,
                        'loss_history': loss_history,
                        'best_loss': best_loss,
                    }
                    safe_save_model(checkpoint_state, checkpoint_path, f"Checkpoint {iteration}")
                
                iteration += 1
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx} in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 如果早期停止，跳出外层循环
        if no_improve_count >= patience:
            break
        
        # Epoch结束统计
        if epoch_batches > 0:
            epoch_avg_loss = epoch_loss / epoch_batches
            print(f"📊 Epoch {epoch} completed:")
            print(f"  • Batches processed: {epoch_batches}")
            print(f"  • Average loss: {epoch_avg_loss:.6f}")
            print(f"  • Total iterations so far: {iteration}")
        else:
            print(f"⚠️ Epoch {epoch}: No valid batches processed")
            break
    
    # 保存最终模型
    print(f"\n🏁 Training completed!")
    print(f"  • Total epochs: {epoch}")
    print(f"  • Total iterations: {iteration}")
    print(f"  • Best loss: {best_loss:.6f}")
    print(f"  • Early stopped: {no_improve_count >= patience}")
    
    final_model_state = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
        'best_loss': best_loss,
        'final_loss': loss_history[-1] if loss_history else float('inf'),
        'total_epochs': epoch,
        'total_iterations': iteration,
        'early_stopped': no_improve_count >= patience,
        'params': params.__dict__ if hasattr(params, '__dict__') else params,
    }
    
    if safe_save_model(final_model_state, final_model_path, "Final model"):
        print(f"🏆 Best loss achieved: {best_loss:.6f}")
    
    # 绘制训练曲线
    if len(loss_history) > 10:
        try:
            plot_training_loss(loss_history, checkpoint_dir)
        except Exception as e:
            print(f"⚠️ Could not save loss plot: {e}")
    
    print("✅ Training process completed successfully!")

def plot_training_loss(loss_history, save_dir, prefix=""):
    """绘制训练损失曲线"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # 原始损失
        plt.subplot(2, 3, 1)
        plt.plot(loss_history, alpha=0.7, linewidth=1)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # 对数尺度损失
        plt.subplot(2, 3, 2)
        plt.plot(loss_history, alpha=0.7, linewidth=1)
        plt.title('Training Loss (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 滑动平均
        if len(loss_history) > 50:
            window_size = min(50, len(loss_history) // 10)
            smoothed = []
            for i in range(window_size, len(loss_history)):
                smoothed.append(sum(loss_history[i-window_size:i]) / window_size)
            
            plt.subplot(2, 3, 3)
            plt.plot(range(window_size, len(loss_history)), smoothed, color='red', alpha=0.8, linewidth=2)
            plt.title(f'Smoothed Loss (window={window_size})')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        # 最近的损失
        if len(loss_history) > 100:
            plt.subplot(2, 3, 4)
            recent_loss = loss_history[-100:]
            plt.plot(range(len(loss_history)-100, len(loss_history)), recent_loss, color='green', alpha=0.8, linewidth=1.5)
            plt.title('Recent Loss (Last 100 iterations)')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        # 损失变化率
        if len(loss_history) > 10:
            loss_changes = [loss_history[i] - loss_history[i-1] for i in range(1, len(loss_history))]
            plt.subplot(2, 3, 5)
            plt.plot(loss_changes, alpha=0.6, color='purple', linewidth=1)
            plt.title('Loss Change Rate')
            plt.xlabel('Iteration')
            plt.ylabel('Loss Δ')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 统计信息
        plt.subplot(2, 3, 6)
        stats_text = f'''Training Statistics

Total Iterations: {len(loss_history)}
Final Loss: {loss_history[-1]:.6f}
Best Loss: {min(loss_history):.6f}
Average Loss: {sum(loss_history)/len(loss_history):.6f}'''
        
        if len(loss_history) > 1:
            improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
            stats_text += f'\nTotal Improvement: {improvement:.2f}%'
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        plt.title('Training Statistics')
        plt.axis('off')
        
        plt.tight_layout()
        loss_plot_path = save_dir / f'{prefix}training_loss_analysis.png'
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training loss analysis saved: {loss_plot_path}")
        
    except ImportError:
        print("⚠️ matplotlib not available, skipping loss plot")
    except Exception as e:
        print(f"⚠️ Error creating loss plot: {e}")

def train_distributed(replica_id, replica_count, port, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    backend = 'nccl' if (torch.cuda.is_available() and os.name != 'nt') else 'gloo'

    if torch.cuda.is_available():
        torch.cuda.set_device(replica_id)

    torch.distributed.init_process_group(
        backend=backend, rank=replica_id, world_size=replica_count
    )

    dataset = from_path(params, is_distributed=True)

    device = torch.device(f'cuda:{replica_id}' if torch.cuda.is_available() else 'cpu')
    
    # 根据是否启用mask选择模型
    use_mask = getattr(params, 'use_mask', False) or getattr(params, 'prediction_mode', False)
    if use_mask and MASK_MODEL_AVAILABLE:
        model = tfdiff_WiFi(params).to(device)
    else:
        model = tfdiff_WiFi(params).to(device)
    
    model = DistributedDataParallel(
        model,
        device_ids=[replica_id] if device.type == 'cuda' else None,
        output_device=replica_id if device.type == 'cuda' else None
    )
    _train_impl(replica_id, model, dataset, params)

def main(args):
    if args.task_id == 4:
        from params import AttrDict
        import numpy as np
        
        # 确保预测长度合理
        total_seq_len = 168
        pred_len = min(args.pred_len, total_seq_len // 2)  # 最大不超过总长度的一半
        input_seq_len = total_seq_len - pred_len
        
        if input_seq_len < 10:  # 确保输入序列足够长
            pred_len = total_seq_len - 10
            input_seq_len = 10
            print(f"⚠️ Adjusted pred_len to {pred_len} to ensure minimum input length")
        
        params = AttrDict(
            task_id=4,
            log_dir='./log/traffic_prediction',
            model_dir='./model/traffic_prediction',
            data_dir=['./dataset/traffic'],
            traffic_path='traffic_data_new.npz',
            embedding_path='environment_embeddings.npz',
            max_iter=args.max_iter or 50000,  # 默认1000次迭代
            batch_size=args.batch_size or 64,
            learning_rate=1e-4,
            max_grad_norm=0.5,
            inference_batch_size=4,
            robust_sampling=True,
            
            # 预测任务相关参数
            pred_len=pred_len,
            seq_len=input_seq_len,
            input_seq_len=input_seq_len,
            total_seq_len=total_seq_len,
            
            # 模型参数
            input_dim=20,
            output_dim=20,
            extra_dim=[128],
            cond_dim=148,
            embed_dim=128,
            hidden_dim=128,
            num_heads=4,
            num_block=8,
            dropout=0.1,
            mlp_ratio=4.0,
            
            # 扩散模型参数
            learn_tfdiff=False,
            max_step=1000,
            signal_diffusion=True,
            blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
            noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
            device=args.device or 'cuda',
            
            prediction_mode=True,
            sample_rate=20,
            
            # Mask相关参数（新增）
            use_mask=True,  # 启用mask模式
            mask_length=pred_len,  # mask长度等于预测长度
            mask_strategy=getattr(args, 'mask_strategy', 'prefix'),  # mask策略
            mask_strategies=['prefix'],  # 使用前缀mask策略
            mask_weight=1.0,  # mask区域损失权重
            unmask_weight=0.1,  # 非mask区域损失权重
            
            # 早期停止参数
            early_stopping_patience=200,  # 早期停止的耐心值
        )
        
        print(f"📊 Prediction Configuration:")
        print(f"  • Total sequence length: {total_seq_len}")
        print(f"  • Input sequence length: {input_seq_len}")
        print(f"  • Prediction length: {pred_len}")
        print(f"  • Mask length: {params.mask_length}")
        print(f"  • Mask strategy: {params.mask_strategy}")
        print(f"  • Use mask: {params.use_mask}")
        print(f"  • Early stopping patience: {params.early_stopping_patience}")
        print(f"  • Validation: {input_seq_len} + {pred_len} = {input_seq_len + pred_len} <= {total_seq_len}")
        
        # 参数验证
        assert input_seq_len + pred_len == total_seq_len, f"序列长度配置不一致: {input_seq_len} + {pred_len} != {total_seq_len}"
        assert params.mask_length <= pred_len, f"mask长度({params.mask_length})不能超过预测长度({pred_len})"
        assert input_seq_len >= 10, f"输入序列长度({input_seq_len})至少需要10个时间步"
        
    else:
        params = all_params[args.task_id]
        if args.pred_len is not None:
            params.pred_len = args.pred_len
            params.prediction_mode = True
            # 启用mask模式
            params.use_mask = True
            params.mask_length = args.pred_len
            params.mask_strategy = getattr(args, 'mask_strategy', 'prefix')
            if hasattr(params, 'seq_len'):
                params.input_seq_len = params.seq_len - args.pred_len
                params.seq_len = params.input_seq_len
    
    # 命令行参数覆盖
    if args.batch_size is not None:
        params.batch_size = args.batch_size
    if args.model_dir is not None:
        params.model_dir = args.model_dir
    if args.data_dir is not None:
        params.data_dir = args.data_dir
    if args.log_dir is not None:
        params.log_dir = args.log_dir
    if args.max_iter is not None:
        params.max_iter = args.max_iter
    
    # 新增mask相关参数处理
    if hasattr(args, 'mask_length') and args.mask_length is not None:
        params.mask_length = args.mask_length
        params.use_mask = True
    
    if hasattr(args, 'mask_strategy') and args.mask_strategy is not None:
        params.mask_strategy = args.mask_strategy
    
    if hasattr(args, 'use_mask') and args.use_mask:
        params.use_mask = True
    
    # 添加早期停止参数
    if not hasattr(params, 'early_stopping_patience'):
        params.early_stopping_patience = 200
    
    # 验证预测长度参数
    if hasattr(params, 'prediction_mode') and params.prediction_mode:
        if params.pred_len <= 0:
            raise ValueError(f"Prediction length must be positive, got {params.pred_len}")
        if hasattr(params, 'total_seq_len') and params.pred_len >= params.total_seq_len:
            raise ValueError(f"Prediction length ({params.pred_len}) must be less than total sequence length ({params.total_seq_len})")
        
        # 验证mask配置
        if params.use_mask:
            if params.mask_length <= 0:
                raise ValueError(f"Mask length must be positive, got {params.mask_length}")
            if params.mask_length > params.pred_len:
                print(f"⚠️ Warning: mask_length ({params.mask_length}) > pred_len ({params.pred_len}), adjusting...")
                params.mask_length = params.pred_len
    
    print(f"\n🎯 Final Configuration Summary:")
    print(f"  • Task ID: {params.task_id}")
    print(f"  • Max iterations: {params.max_iter}")
    print(f"  • Batch size: {params.batch_size}")
    print(f"  • Learning rate: {getattr(params, 'learning_rate', 'N/A')}")
    print(f"  • Use mask: {getattr(params, 'use_mask', False)}")
    if hasattr(params, 'use_mask') and params.use_mask:
        print(f"  • Mask length: {getattr(params, 'mask_length', 'N/A')}")
        print(f"  • Mask strategy: {getattr(params, 'mask_strategy', 'N/A')}")
    print(f"  • Early stopping patience: {getattr(params, 'early_stopping_patience', 'N/A')}")
    
    replica_count = device_count()
    
    if replica_count > 1 and getattr(params, 'device', 'cuda') != 'cpu':
        if params.batch_size % replica_count != 0:
            raise ValueError(
                f'Batch size {params.batch_size} is not evenly divisible by # GPUs {replica_count}.')
        params.batch_size = params.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, params), nprocs=replica_count, join=True)
    else:
        train(params)

if __name__ == '__main__':
    parser = ArgumentParser(
        description='train (or resume training) a tfdiff prediction model with mask support')
    parser.add_argument('--task_id', default=4, type=int,
                        help='use case of tfdiff model, 0/1/2/3/4 for WiFi/FMCW/MIMO/EEG/Traffic respectively')
    parser.add_argument('--pred_len', default=10, type=int,
                        help='prediction length (l parameter), number of time steps to predict')
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data_dir', default=None, nargs='+',
                        help='space separated list of directories from which to read data files for training')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--max_iter', default=50000, type=int,
                        help='maximum number of training iteration')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--device', default=None, type=str,
                        help='device to use for training (cuda/cpu)')
    
    # 新增mask相关参数
    parser.add_argument('--mask_length', default=None, type=int,
                        help='length of mask (number of time steps to mask)')
    parser.add_argument('--use_mask', action='store_true',
                        help='enable mask-aware training')
    parser.add_argument('--mask_strategy', default='prefix', choices=['prefix', 'suffix', 'random'],
                        help='strategy for creating masks')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.pred_len <= 0:
        raise ValueError("--pred_len must be a positive integer")
    
    if args.mask_length is not None and args.mask_length <= 0:
        raise ValueError("--mask_length must be a positive integer")
    
    # 如果指定了mask_length，自动启用mask模式
    if args.mask_length is not None:
        args.use_mask = True
    
    # 如果指定了pred_len，自动启用mask模式（除非明确禁用）
    if args.pred_len is not None and not hasattr(args, 'no_mask'):
        args.use_mask = True
        if args.mask_length is None:
            args.mask_length = args.pred_len
    
    print(f"🚀 Starting TFDiff training...")
    print(f"📋 Arguments: {vars(args)}")
    
    main(args)
