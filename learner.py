import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusion import SignalDiffusion, GaussianDiffusion
from dataset import _nested_map

device = torch.device('cuda')

import torch
import torch.nn as nn

class tfdiffLoss(nn.Module):
    def __init__(self, w=0.05, lambda_cosine=0.3):
        super().__init__()
        self.w = w
        self.lambda_cosine = lambda_cosine  # 重命名为更准确的参数名
        self.eps = 1e-8

    def forward(self, target, est, target_noise=None, est_noise=None):
        # 确保输入是有限的
        if not torch.isfinite(target).all() or not torch.isfinite(est).all():
            print("⚠️ 损失函数输入包含非有限值")
            return torch.tensor(0.0, device=target.device, requires_grad=True)
        
        # 计算基本的MSE损失，添加数值稳定性
        t_loss = self.stable_mse_loss(target, est)
        
        # 计算噪声损失
        if target_noise is not None and est_noise is not None:
            if torch.isfinite(target_noise).all() and torch.isfinite(est_noise).all():
                n_loss = self.stable_mse_loss(target_noise, est_noise)
            else:
                n_loss = 0.0
        else:
            n_loss = 0.0
        
        # 计算余弦相似度损失
        try:
            cosine_loss = self.cosine_similarity_loss(target, est)
        except Exception as e:
            print(f"余弦相似度损失计算出错: {e}")
            cosine_loss = 0.0
        
        # 总损失 - 注意这里使用加法，因为我们要最小化(1-cosine_similarity)
        total_loss = t_loss + self.w * n_loss + self.lambda_cosine * cosine_loss
        
        # 最终检查
        if not torch.isfinite(total_loss).all():
            print("⚠️ 总损失包含非有限值")
            return torch.tensor(0.0, device=target.device, requires_grad=True)
        
        # 裁剪损失值避免过大
        total_loss = torch.clamp(total_loss, 0.0, 1e6)
        
        return total_loss

    def stable_mse_loss(self, target, est):
        """数值稳定的MSE损失"""
        diff = target - est
        # 裁剪差值以避免过大的值
        diff = torch.clamp(diff, -1e3, 1e3)
        return torch.mean(diff ** 2)

    def cosine_similarity_loss(self, target, est):
        """
        计算余弦相似度损失
        
        Args:
            target: 真实值矩阵 X_true ∈ R^{T×C}
            est: 预测值矩阵 X̂ ∈ R^{T×C}
            
        Returns:
            余弦相似度损失 (1 - cosine_similarity)
        """
        batch_size = target.shape[0]
        cosine_losses = []
        
        for i in range(batch_size):
            # 获取单个样本的目标和预测矩阵
            target_sample = target[i]  # shape: [T, C] 或其他维度
            est_sample = est[i]        # shape: [T, C] 或其他维度
            
            # 将矩阵展平为向量 vec(X)
            a = target_sample.flatten()  # vec(X_true)
            b = est_sample.flatten()     # vec(X̂)
            
            # 计算向量的L2范数
            norm_a = torch.sqrt(torch.sum(a**2) + self.eps)
            norm_b = torch.sqrt(torch.sum(b**2) + self.eps)
            
            # 计算余弦相似度: cos(θ) = (a·b) / (||a|| * ||b||)
            dot_product = torch.sum(a * b)
            cosine_similarity = dot_product / (norm_a * norm_b + self.eps)
            
            # 裁剪余弦相似度到合理范围 [-1, 1]
            cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
            
            # 计算余弦相似度损失: 1 - cos(θ)
            # 当向量完全相似时，cos(θ) = 1，损失 = 0
            # 当向量完全不相似时，cos(θ) = -1，损失 = 2
            cosine_loss = 1.0 - cosine_similarity
            cosine_losses.append(cosine_loss)
        
        # 返回批次的平均余弦相似度损失
        return torch.stack(cosine_losses).mean()

    def stable_cross_loss(self, target, est):
        """
        保留原始的交叉损失函数以备需要时使用
        """
        batch_size, num_channels = target.shape[0], target.shape[-1]
        
        try:
            target = target.reshape(batch_size, -1, num_channels)
            est = est.reshape(batch_size, -1, num_channels)
            
            cross_corr = torch.mean(target * est, dim=1)
            target_norm = torch.sqrt(torch.mean(target**2, dim=1) + self.eps)
            est_norm = torch.sqrt(torch.mean(est**2, dim=1) + self.eps)
            
            normalized_cross_corr = cross_corr / (target_norm * est_norm + self.eps)
            # 裁剪结果避免极值
            normalized_cross_corr = torch.clamp(normalized_cross_corr, -1.0, 1.0)
            
            return torch.mean(normalized_cross_corr)
        except Exception as e:
            print(f"交叉损失计算出错: {e}")
            return torch.tensor(0.0, device=target.device)


class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
         
        self.model_dir = model_dir
        self.task_id = params.task_id
        self.log_dir = log_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = next(self.model.parameters()).device
        self.diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
        self.diffusion.to(self.device)   # ← 关键
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)  # 更温和的学习率衰减
        self.params = params
        self.iter = 0
        self.is_master = True
        self.loss_fn = tfdiffLoss()
        self.summary_writer = None
        self.skip_count = 0  # 跳过的批次计数
        self.max_skip_ratio = 0.5  # 最大跳过比例
        # 为本次运行创建唯一子目录
        self.run_id = f"run_{int(time.time())}_{os.getpid()}"
        self.run_log_dir = os.path.join(self.log_dir, self.run_id)
        os.makedirs(self.run_log_dir, exist_ok=True)
        # 初始化模型权重
        self._initialize_model_weights()

    def _initialize_model_weights(self):
        """安全的模型权重初始化"""
        print("🔧 初始化模型权重...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier正态分布初始化，使用更小的增益
                nn.init.xavier_normal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Kaiming正态分布初始化，使用更小的增益
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(module.weight, 'data'):
                    module.weight.data *= 0.1  # 进一步缩小初始权重
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
        
        print("✅ 模型权重初始化完成")

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        save_basename = f'{filename}-{timestamp}.pt'
        save_name = os.path.join(self.model_dir, save_basename)
        link_name = os.path.join(self.model_dir, f'{filename}.pt')

        # 确保保存路径存在
        os.makedirs(self.model_dir, exist_ok=True)

        # 保存模型检查点
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            # 如果存在软链接，删除旧链接并更新
            if os.path.exists(link_name):
                os.remove(link_name)
            os.rename(save_name, link_name)

    def restore_from_checkpoint(self, filename='weights', replica_id=None):
        checkpoint_path = os.path.join(self.model_dir, f'{filename}.pt')
        try:
            # 如果不存在文件，返回错误信息
            if not os.path.exists(checkpoint_path):
                print(f"No checkpoint found at {checkpoint_path}, starting fresh training")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint)
            print(f"✅ Restored from checkpoint at iteration {self.iter}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def _stable_normalize(self, data, eps=1e-8):
        """数值稳定的归一化"""
        # 计算均值和标准差
        dims = tuple(range(1, data.ndim))
        mean = data.mean(dim=dims, keepdim=True)
        std = data.std(dim=dims, keepdim=True)
        
        # 避免除零
        std = torch.clamp(std, min=eps)
        
        # 归一化
        normalized = (data - mean) / std
        
        # 裁剪极值
        normalized = torch.clamp(normalized, -5.0, 5.0)
        
        return normalized
    
    def _reinitialize_parameter(self, param):
        """重新初始化单个参数"""
        with torch.no_grad():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param, gain=0.01)
    
    def _check_and_fix_gradients(self):
        """检查和修复梯度问题"""
        has_valid_grad = False
        total_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 检查梯度是否有限
                if not torch.isfinite(param.grad).all():
                    print(f"⚠️ 参数 {name} 的梯度包含非有限值，清零梯度")
                    param.grad.zero_()
                    continue
                
                # 检查梯度是否过大
                grad_norm = param.grad.data.norm(2).item()
                if grad_norm > 1e2:
                    print(f"⚠️ 参数 {name} 的梯度过大 ({grad_norm:.2e})，进行裁剪")
                    param.grad.data = torch.clamp(param.grad.data, -1e2, 1e2)
                    grad_norm = param.grad.data.norm(2).item()
                
                total_norm += grad_norm ** 2
                has_valid_grad = True
        
        if not has_valid_grad:
            print("⚠️ 没有有效的梯度，跳过当前批次")
            return False
        
        total_norm = total_norm ** 0.5
        if total_norm == 0:
            print("⚠️ 总梯度范数为零，跳过当前批次")
            return False
        
        return True

    def train_iter(self, features):
        """改进的训练迭代函数，处理梯度数值不稳定问题"""
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        try:
            data = features['data']
            cond = features['cond']
        except KeyError as e:
            print(f"❌ Missing key in features: {e}")
            return None
        
        # 1. 更严格的数据检查
        if not torch.isfinite(data).all():
            print("⚠️ 输入数据包含非有限值，跳过当前批次。")
            self.skip_count += 1
            return None
        
        if not torch.isfinite(cond).all():
            print("⚠️ 条件数据包含非有限值，跳过当前批次。")
            self.skip_count += 1
            return None
        
        # 2. 数据范围检查和裁剪
        if data.abs().max() > 1e3:
            print(f"⚠️ 输入数据值过大 (max: {data.abs().max():.2e})，进行裁剪")
            data = torch.clamp(data, -1e3, 1e3)
        
        if cond.abs().max() > 1e3:
            print(f"⚠️ 条件数据值过大 (max: {cond.abs().max():.2e})，进行裁剪")
            cond = torch.clamp(cond, -1e3, 1e3)
        
        # 3. 数据归一化以提高数值稳定性
        data = self._stable_normalize(data)
        cond = self._stable_normalize(cond)
        
        # 4. 检查模型参数是否有问题
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"❌ 模型参数 {name} 包含非有限值! 重新初始化...")
                self._reinitialize_parameter(param)
                return None
        
        B = data.shape[0]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64, device=data.device)
        
        # 5. 扩散过程
        try:
            degrade_data = self.diffusion.degrade_fn(data, t, self.task_id)
        except Exception as e:
            print(f"❌ 扩散过程出错: {e}")
            self.skip_count += 1
            return None
        
        # 检查扩散后的数据
        if not torch.isfinite(degrade_data).all():
            print("⚠️ 扩散后的数据包含非有限值，跳过当前批次。")
            self.skip_count += 1
            return None
        
        # 6. 模型前向传播
        try:
            predicted = self.model(degrade_data, t, cond)
        except Exception as e:
            print(f"❌ 模型前向传播出错: {e}")
            self.skip_count += 1
            return None
        
        # 检查预测结果
        if not torch.isfinite(predicted).all():
            print("⚠️ 预测结果包含非有限值，跳过当前批次。")
            self.skip_count += 1
            return None
        
        # 7. 计算损失
        try:
            loss = self.loss_fn(data, predicted)
        except Exception as e:
            print(f"❌ 损失计算出错: {e}")
            self.skip_count += 1
            return None
        
        # 检查损失值
        if not torch.isfinite(loss).all():
            print(f"⚠️ 损失值异常 ({loss.item()})，跳过当前批次。")
            self.skip_count += 1
            return None
        
        # 检查损失值是否过大
        if loss.item() > 1e3:
            print(f"⚠️ 损失值过大 ({loss.item():.2e})，跳过当前批次。")
            self.skip_count += 1
            return None
        
        # 8. 反向传播
        try:
            loss.backward()
        except Exception as e:
            print(f"❌ 反向传播出错: {e}")
            self.skip_count += 1
            return None
        
        # 9. 检查和处理梯度
        if not self._check_and_fix_gradients():
            self.skip_count += 1
            return None
        
        # 10. 梯度裁剪 - 使用更小的阈值
        max_grad_norm = getattr(self.params, 'max_grad_norm', 0.5)  # 更严格的梯度裁剪
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # 11. 优化器步骤
        try:
            self.optimizer.step()
        except Exception as e:
            print(f"❌ 优化器步骤出错: {e}")
            self.skip_count += 1
            return None
        
        # 12. 后处理检查
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"❌ 优化后参数 {name} 包含非有限值! 重新初始化...")
                self._reinitialize_parameter(param)
                return None
        
        return loss

    def train(self, max_iter=None):
        device = next(self.model.parameters()).device
        epoch = 0
        total_processed_batches = 0
        
        print(f"🚀 Starting training with max_iter={max_iter}")
        print(f"📊 Dataset length: {len(self.dataset)}")
        try:
            while True:  # epoch loop
                epoch_start_iter = self.iter
                epoch_processed_batches = 0
                epoch_skipped_batches = 0
                print(f"\n🔄 Starting Epoch {epoch + 1}")
                
                # 重置跳过计数
                self.skip_count = 0
                
                # 创建进度条
                pbar = tqdm(self.dataset, desc=f'Epoch {epoch + 1}', disable=not self.is_master)
                
                for batch_idx, features in enumerate(pbar):
                    if max_iter is not None and self.iter >= max_iter:
                        print(f"✅ Reached max_iter {max_iter}, stopping training")
                        pbar.close()
                        return
                    
                    # 确保数据移动到正确的设备
                    # train() 内 for 循环里
                    features = _nested_map(features, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)

                    loss = self.train_iter(features)
                    
                    if loss is None:
                        epoch_skipped_batches += 1
                        # 检查跳过比例是否过高
                        if epoch_processed_batches > 0:
                            skip_ratio = epoch_skipped_batches / (epoch_processed_batches + epoch_skipped_batches)
                            if skip_ratio > self.max_skip_ratio:
                                print(f"❌ 跳过比例过高 ({skip_ratio:.2%})，可能存在严重问题")
                                print("   建议检查:")
                                print("   1. 数据预处理是否正确")
                                print("   2. 模型架构是否合适")
                                print("   3. 学习率是否过大")
                                print("   4. 初始化是否合理")
                                # 降低学习率尝试恢复
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] *= 0.5
                                print(f"   已将学习率降低到 {self.optimizer.param_groups[0]['lr']:.2e}")
                        continue

                    # 检查损失是否合理
                    loss_val = loss.item()
                    if loss_val > 1e6:
                        print(f"⚠️ 损失值异常大: {loss_val:.2e}")
                    elif loss_val < 1e-10:
                        print(f"⚠️ 损失值异常小: {loss_val:.2e}")

                    # 更新进度条信息
                    if self.is_master:
                        pbar.set_postfix({
                            'loss': f'{loss_val:.6f}',
                            'iter': self.iter,
                            'batch': epoch_processed_batches + 1,
                            'skipped': epoch_skipped_batches,
                            'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                        })
                        
                        if self.iter % 50 == 0:
                            self._write_summary(self.iter, features, loss)
                            
                    epoch_processed_batches += 1
                    total_processed_batches += 1
                    self.iter += 1
                
                pbar.close()
                
                # Epoch 结束后的处理
                epoch += 1
                skip_ratio = epoch_skipped_batches / max(1, epoch_processed_batches + epoch_skipped_batches)
                
                print(f"✅ Completed Epoch {epoch}")
                print(f"📈 Processed batches: {epoch_processed_batches}")
                print(f"⚠️ Skipped batches: {epoch_skipped_batches} ({skip_ratio:.1%})")
                print(f"📊 Total processed batches so far: {total_processed_batches}")
                
                # 学习率调度
                old_lr = self.optimizer.param_groups[0]['lr']
                self.lr_scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f"📉 Learning rate updated: {old_lr:.2e} -> {new_lr:.2e}")
                
                # 每个 epoch 结束后保存检查点
                if self.is_master:
                    self.save_to_checkpoint()
                    print(f"💾 Checkpoint saved at iteration {self.iter}")
                    
                # 如果达到最大迭代次数，退出
                if max_iter is not None and self.iter >= max_iter:
                    print(f"🎉 Training completed after {epoch} epochs and {self.iter} iterations")
                    return
                
                # 检查是否处理了足够的批次
                if epoch_processed_batches == 0:
                    print("❌ Warning: No batches were processed in this epoch!")
                    print("   This might indicate a serious problem with the dataset or model.")
                    print("   尝试以下解决方案:")
                    print("   1. 检查数据范围和分布")
                    print("   2. 降低学习率")
                    print("   3. 重新初始化模型")
                    
                    # 尝试恢复措施
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    print(f"   学习率已降低到 {self.optimizer.param_groups[0]['lr']:.2e}")
                    
                    # 重新初始化模型权重
                    self._initialize_model_weights()
                    print("   模型权重已重新初始化")
                    
                    if epoch > 3:  # 如果连续多个epoch都无法处理，则停止
                        print("❌ 连续多个epoch无法处理批次，停止训练")
                        break
        finally:
            if self.summary_writer is not None:
                try:
                    self.summary_writer.flush()
                    self.summary_writer.close()
                except Exception:
                    pass

    def _write_summary(self, iter, features, loss):
        # 只初始化一次；指定唯一后缀，减少冲突；适度队列+自动刷盘
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(
                self.run_log_dir,
                purge_step=iter,
                filename_suffix=f".{os.getpid()}",
                max_queue=50,
                flush_secs=10,
            )

        try:
            loss_scalar = loss.item() if torch.is_tensor(loss) else float(loss)
            self.summary_writer.add_scalar('train/loss', loss_scalar, iter)
            if hasattr(self, "grad_norm"):
                gn = self.grad_norm.item() if torch.is_tensor(self.grad_norm) else float(self.grad_norm)
                self.summary_writer.add_scalar('train/grad_norm', gn, iter)
            self.summary_writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], iter)
            self.summary_writer.flush()
        except PermissionError as e:
            print(f"[TensorBoard] 写入失败：{e}，切换新日志目录继续写入。")
            # 兜底：换新目录重开 writer
            try:
                self.summary_writer.close()
            except Exception:
                pass
            reopen = os.path.join(self.log_dir, f"reopen_{int(time.time())}_{os.getpid()}")
            os.makedirs(reopen, exist_ok=True)
            self.summary_writer = SummaryWriter(reopen, purge_step=iter, filename_suffix=f".{os.getpid()}")
