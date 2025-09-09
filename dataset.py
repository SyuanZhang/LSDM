import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import scipy.io as scio
from params import AttrDict
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path


class TrafficPredictionDataset(Dataset):
    def __init__(self, traffic_path="traffic_data.npz", embedding_path="environment_embeddings.npz",
                 mode='train', train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                 hist_len=148, pred_len=20, stride=1):
        """
        流量预测数据集 - 按用户划分版本
        """
        super().__init__()
        
        self.mode = mode
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.stride = stride
        
        print(f"🏗️ 初始化流量预测数据集 ({mode} 模式)")
        print(f"📋 配置:")
        print(f"  • 历史长度: {hist_len}")
        print(f"  • 预测长度: {pred_len}")
        print(f"  • 用户划分比例: {train_ratio:.1f}:{val_ratio:.1f}:{test_ratio:.1f}")
        
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("用户划分比例之和必须为1.0")
        
        # 加载数据
        self._load_data(traffic_path, embedding_path)
        
        # 按用户划分
        self._split_users(train_ratio, val_ratio, test_ratio)
        
        # 生成时间窗口样本
        self._generate_samples()
        
        print(f"✅ {mode} 数据集初始化完成:")
        print(f"  • 分配用户数: {len(self.user_ids)}")
        print(f"  • 生成样本数: {len(self.samples)}")
    
    def _load_data(self, traffic_path, embedding_path):
        """加载流量和环境数据"""
        if not os.path.exists(traffic_path):
            self._create_sample_data(traffic_path, embedding_path)
        
        # 加载流量数据 [n_users, n_apps, n_timesteps]
        traffic_data = np.load(traffic_path)
        self.traffic_matrix = traffic_data['traffic_matrix']
        
        # 加载环境嵌入 [n_users, n_timesteps, embedding_dim]
        if os.path.exists(embedding_path):
            env_data = np.load(embedding_path)
            self.env_embeddings = env_data['embeddings']
        else:
            # 如果环境嵌入文件不存在，创建随机嵌入
            print(f"⚠️ 环境嵌入文件 {embedding_path} 不存在，创建随机嵌入")
            self.env_embeddings = np.random.randn(
                self.traffic_matrix.shape[0], 
                self.traffic_matrix.shape[2], 
                128
            )
        
        self.n_users, self.n_apps, self.n_timesteps = self.traffic_matrix.shape
        self.embedding_dim = self.env_embeddings.shape[2]
        
        print(f"📊 数据维度:")
        print(f"  • 总用户数: {self.n_users}")
        print(f"  • 应用数量: {self.n_apps}")
        print(f"  • 时间步数: {self.n_timesteps}")
        print(f"  • 嵌入维度: {self.embedding_dim}")
        
        # 验证时间维度
        total_required = self.hist_len + self.pred_len
        if total_required > self.n_timesteps:
            raise ValueError(
                f"所需时间长度 ({total_required} = {self.hist_len}+{self.pred_len}) "
                f"> 可用时间步数 ({self.n_timesteps})\n"
                f"请减少 hist_len 或 pred_len"
            )
    
    def _split_users(self, train_ratio, val_ratio, test_ratio):
        """按用户维度划分数据集 - 确保无重叠"""
        print(f"👥 按用户划分数据集...")
        
        # 设置固定随机种子确保可重现性
        np.random.seed(42)
        all_user_ids = np.arange(self.n_users)
        np.random.shuffle(all_user_ids)
        
        # 计算精确的划分点
        train_end = int(np.floor(self.n_users * train_ratio))
        val_end = train_end + int(np.floor(self.n_users * val_ratio))
        
        # 确保所有用户都被分配
        if val_end >= self.n_users:
            val_end = self.n_users - 1
        
        # 按模式选择用户
        if self.mode == 'train':
            self.user_ids = all_user_ids[:train_end]
        elif self.mode == 'val':
            self.user_ids = all_user_ids[train_end:val_end]
        elif self.mode == 'test':
            self.user_ids = all_user_ids[val_end:]
        else:
            raise ValueError(f"未知模式: {self.mode}")
        
        # 确保每个集合至少有一个用户
        if len(self.user_ids) == 0:
            if self.mode == 'train':
                self.user_ids = all_user_ids[:max(1, self.n_users//2)]
            elif self.mode == 'val':
                start_idx = max(1, self.n_users//2)
                self.user_ids = all_user_ids[start_idx:start_idx+max(1, self.n_users//4)]
            else:  # test
                self.user_ids = all_user_ids[-max(1, self.n_users//4):]
        
        print(f"📊 {self.mode} 集合用户分配:")
        print(f"  • 用户数量: {len(self.user_ids)}")
        print(f"  • 用户ID范围: [{self.user_ids.min()}, {self.user_ids.max()}]")
        print(f"  • 占总用户比例: {len(self.user_ids)/self.n_users:.1%}")
    
    def _generate_samples(self):
        """为分配的用户生成滑动窗口样本"""
        print(f"🔄 为 {len(self.user_ids)} 个用户生成样本...")
        
        self.samples = []
        
        # 计算时间窗口范围
        max_start_time = self.n_timesteps - self.hist_len - self.pred_len
        
        if max_start_time < 0:
            raise ValueError("时间维度不足以生成样本")
        
        print(f"📅 时间窗口:")
        print(f"  • 最大起始时间: {max_start_time}")
        print(f"  • 滑动步长: {self.stride}")
        
        total_samples = 0
        for user_id in self.user_ids:
            user_samples = 0
            
            # 为每个用户生成滑动窗口样本
            for start_time in range(0, max_start_time + 1, self.stride):
                hist_start = start_time
                hist_end = start_time + self.hist_len
                pred_start = hist_end
                pred_end = pred_start + self.pred_len
                
                if pred_end <= self.n_timesteps:
                    self.samples.append({
                        'user_id': user_id,
                        'hist_start': hist_start,
                        'hist_end': hist_end,
                        'pred_start': pred_start,
                        'pred_end': pred_end
                    })
                    user_samples += 1
                    total_samples += 1
            
            # 显示第一个用户的详细信息
            if user_id == self.user_ids[0]:
                print(f"  • 用户 {user_id} 生成 {user_samples} 个样本")
        
        print(f"  • 总样本数: {total_samples}")
        print(f"  • 平均每用户: {total_samples/len(self.user_ids):.1f} 样本")
        
        if len(self.samples) == 0:
            raise ValueError("无法生成任何有效样本")
    
    def _create_sample_data(self, traffic_path, embedding_path):
        """创建示例数据"""
        print("🔧 创建示例流量数据...")
        np.random.seed(42)
        
        # 创建更真实的流量数据
        n_users, n_apps, n_timesteps = 871, 20, 168
        
        # 生成带周期性的流量数据
        traffic_matrix = np.zeros((n_users, n_apps, n_timesteps))
        
        for user in range(n_users):
            for app in range(n_apps):
                # 基础流量水平
                base_traffic = np.random.exponential(1.0)
                
                # 添加日周期性 (24小时周期)
                hourly_pattern = np.sin(2 * np.pi * np.arange(n_timesteps) / 24) + 1
                
                # 添加周周期性 (168小时=7天周期)
                weekly_pattern = 0.5 * np.cos(2 * np.pi * np.arange(n_timesteps) / 168) + 1
                
                # 添加随机噪声
                noise = np.random.normal(0, 0.3, n_timesteps)
                
                # 组合所有成分
                traffic_matrix[user, app, :] = (
                    base_traffic * hourly_pattern * weekly_pattern + noise
                ).clip(min=0)
        
        # 确保目录存在
        Path(traffic_path).parent.mkdir(parents=True, exist_ok=True)
        Path(embedding_path).parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(traffic_path, traffic_matrix=traffic_matrix)
        
        # 创建环境嵌入
        env_embeddings = np.random.randn(n_users, n_timesteps, 128)
        np.savez(embedding_path, embeddings=env_embeddings)
        
        print("✅ 示例数据创建完成")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        
        # 提取历史数据 [hist_len, n_apps]
        hist_traffic = self.traffic_matrix[
            user_id, :, sample['hist_start']:sample['hist_end']
        ].T
        
        # 提取预测目标 [pred_len, n_apps]
        target_traffic = self.traffic_matrix[
            user_id, :, sample['pred_start']:sample['pred_end']
        ].T
        
        # 提取环境嵌入
        hist_env = self.env_embeddings[
            user_id, sample['hist_start']:sample['hist_end'], :
        ]
        
        pred_env = self.env_embeddings[
            user_id, sample['pred_start']:sample['pred_end'], :
        ]
        
        return {
            'hist_traffic': torch.from_numpy(hist_traffic).float(),
            'target_traffic': torch.from_numpy(target_traffic).float(),
            'hist_env': torch.from_numpy(hist_env).float(),
            'pred_env': torch.from_numpy(pred_env).float(),
            'user_id': user_id,
            'time_info': {
                'hist_range': (sample['hist_start'], sample['hist_end']),
                'pred_range': (sample['pred_start'], sample['pred_end']),
                'mode': self.mode
            }
        }


class TrafficCollator:
    """流量数据的批处理整理器"""
    
    def __init__(self, normalize=True, noise_std=0.01, task_params=None):
        self.normalize = normalize
        self.noise_std = noise_std
        self.task_params = task_params
    
    def collate(self, batch):
        """批处理数据整理 - 修复版本"""
        if len(batch) == 0:
            raise ValueError("Empty batch received!")
        
        batch_size = len(batch)
        
        # 提取数据
        hist_traffic = torch.stack([item['hist_traffic'] for item in batch])  # (B, hist_len, n_apps)
        target_traffic = torch.stack([item['target_traffic'] for item in batch])  # (B, pred_len, n_apps)
        hist_env = torch.stack([item['hist_env'] for item in batch])  # (B, hist_len, embedding_dim)
        pred_env = torch.stack([item['pred_env'] for item in batch])  # (B, pred_len, embedding_dim)
        
        # print(f"🔍 原始数据形状:")
        # print(f"  • hist_traffic: {hist_traffic.shape}")
        # print(f"  • target_traffic: {target_traffic.shape}")
        # print(f"  • hist_env: {hist_env.shape}")
        
        # 数据标准化
        if self.normalize:
            hist_mean = hist_traffic.mean(dim=(1, 2), keepdim=True)
            hist_std = hist_traffic.std(dim=(1, 2), keepdim=True) + 1e-8
            
            hist_traffic_norm = (hist_traffic - hist_mean) / hist_std
            target_traffic_norm = (target_traffic - hist_mean) / hist_std
            
            if self.noise_std > 0:
                noise = torch.randn_like(hist_traffic_norm) * self.noise_std
                hist_traffic_norm = hist_traffic_norm + noise
            
            hist_traffic_norm = torch.clamp(hist_traffic_norm, -10, 10)
            target_traffic_norm = torch.clamp(target_traffic_norm, -10, 10)
        else:
            hist_traffic_norm = hist_traffic
            target_traffic_norm = target_traffic
            hist_mean = torch.zeros_like(hist_traffic.mean(dim=1, keepdim=True))
            hist_std = torch.ones_like(hist_traffic.std(dim=1, keepdim=True))
        
        # 🔧 修复：正确的数据格式
        # 目标：让时间维度在正确的位置
        
        # 方案A: 保持时间维度在第二个位置 (推荐)
        # data: (B, time=pred_len, features=n_apps, 1)
        data = target_traffic_norm.unsqueeze(-1)  # (B, pred_len, n_apps, 1)
        
        # 条件数据：组合历史流量和环境嵌入
        if hist_traffic_norm.shape[1] != hist_env.shape[1]:
            min_seq_len = min(hist_traffic_norm.shape[1], hist_env.shape[1])
            hist_traffic_norm = hist_traffic_norm[:, :min_seq_len, :]
            hist_env = hist_env[:, :min_seq_len, :]
        
        # cond: (B, time=hist_len, features=n_apps+embedding_dim)
        combined_cond = torch.cat([hist_traffic_norm, hist_env], dim=-1)
        cond = combined_cond  # 保持 (B, hist_len, features) 格式
        
        # print(f"🔧 修复后数据形状:")
        # print(f"  • data: {data.shape} (B, pred_len={data.shape[1]}, n_apps={data.shape[2]}, 1)")
        # print(f"  • cond: {cond.shape} (B, hist_len={cond.shape[1]}, features={cond.shape[2]})")
        
        return {
            'data': data,  # (B, pred_len, n_apps, 1)
            'cond': cond,  # (B, hist_len, n_apps + embedding_dim)
            'pred_env': pred_env,
            'hist_traffic': hist_traffic,
            'target_traffic': target_traffic,
            'stats': {
                'mean': hist_mean,
                'std': hist_std
            },
            'user_ids': [item['user_id'] for item in batch],
            'time_info': [item['time_info'] for item in batch]
        }


def create_traffic_dataloaders_by_user(traffic_path="traffic_data.npz", 
                                      embedding_path="environment_embeddings.npz",
                                      batch_size=32, hist_len=148, pred_len=20, stride=1,
                                      train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                                      num_workers=0, is_distributed=False):
    """
    创建按用户划分的流量预测数据加载器
    """
    print("🏗️ 创建按用户划分的流量预测数据加载器...")
    print(f"📋 参数配置:")
    print(f"  • 用户划分比例: {train_ratio:.1f}:{val_ratio:.1f}:{test_ratio:.1f}")
    print(f"  • 历史长度: {hist_len}")
    print(f"  • 预测长度: {pred_len}")
    print(f"  • 批次大小: {batch_size}")
    
    # 创建三个数据集
    datasets = {}
    for mode in ['train', 'val', 'test']:
        datasets[mode] = TrafficPredictionDataset(
            traffic_path=traffic_path,
            embedding_path=embedding_path,
            mode=mode,
            train_ratio=train_ratio,
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            hist_len=hist_len,
            pred_len=pred_len,
            stride=stride
        )
    
    # 验证用户不重叠
    _validate_user_split(datasets)
    
    # 创建数据整理器
    collator = TrafficCollator(normalize=True, noise_std=0.01)
    
    # 创建数据加载器
    dataloaders = {}
    for mode in ['train', 'val', 'test']:
        sampler = None
        shuffle = (mode == 'train')
        
        if is_distributed and mode == 'train':
            sampler = DistributedSampler(datasets[mode])
            shuffle = False
        
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=min(batch_size, len(datasets[mode])),
            collate_fn=collator.collate,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    print(f"✅ 数据加载器创建完成!")
    _print_dataset_summary(datasets, dataloaders)
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']


def _validate_user_split(datasets):
    """验证用户划分没有重叠"""
    print("🔍 验证用户划分...")
    
    train_users = set(datasets['train'].user_ids)
    val_users = set(datasets['val'].user_ids)
    test_users = set(datasets['test'].user_ids)
    
    # 检查重叠
    overlaps = {
        'train_val': train_users & val_users,
        'train_test': train_users & test_users,
        'val_test': val_users & test_users
    }
    
    has_overlap = False
    for key, overlap in overlaps.items():
        if overlap:
            print(f"❌ {key} 用户重叠: {len(overlap)} 个用户")
            has_overlap = True
    
    if not has_overlap:
        print("✅ 验证通过: 用户划分完全无重叠")
    
    # 显示统计信息
    total_users = len(train_users) + len(val_users) + len(test_users)
    print(f"📊 用户划分统计:")
    print(f"  • 训练用户: {len(train_users)} ({len(train_users)/total_users:.1%})")
    print(f"  • 验证用户: {len(val_users)} ({len(val_users)/total_users:.1%})")
    print(f"  • 测试用户: {len(test_users)} ({len(test_users)/total_users:.1%})")


def _print_dataset_summary(datasets, dataloaders):
    """打印数据集摘要"""
    print(f"📊 数据集详细统计:")
    for mode in ['train', 'val', 'test']:
        dataset = datasets[mode]
        loader = dataloaders[mode]
        print(f"  • {mode.upper()}:")
        print(f"    - 用户数: {len(dataset.user_ids)}")
        print(f"    - 样本数: {len(dataset)}")
        print(f"    - 批次数: {len(loader)}")
        print(f"    - 每批样本: {loader.batch_size}")


def analyze_traffic_distribution(traffic_data):
    """分析流量数据分布"""
    print("\n" + "="*50)
    print("📊 Traffic Data Analysis")
    print("="*50)
    
    print(f"Shape: {traffic_data.shape}")
    print(f"Mean: {traffic_data.mean():.4f}")
    print(f"Std: {traffic_data.std():.4f}")
    print(f"Min: {traffic_data.min():.4f}")
    print(f"Max: {traffic_data.max():.4f}")
    
    # 按应用类别分析
    app_names = [
        'Social Media', 'Video Streaming', 'Gaming', 'Web Browsing', 'Email',
        'Music Streaming', 'Navigation', 'Shopping', 'News', 'Weather',
        'Productivity', 'Education', 'Health', 'Finance', 'Travel',
        'Photography', 'Communication', 'Entertainment', 'Utilities', 'Others'
    ]
    
    print("\nPer-app statistics:")
    for i, app_name in enumerate(app_names[:min(len(app_names), traffic_data.shape[1])]):
        app_data = traffic_data[:, i, :]
        print(f"  {app_name}: mean={app_data.mean():.4f}, std={app_data.std():.4f}")


# 其他数据集类保持不变
class WiFiDataset(torch.utils.data.Dataset):
    def __init__(self, paths, embedding_path="env_info.npz", target_shape=(168, 10)):
        """
        paths: 数据文件夹的路径列表
        embedding_path: 存储 embedding 的 npz 文件路径
        target_shape: 数据目标形状
        """
        super().__init__()
        self.filenames = []
        self.target_shape = target_shape  # 设置目标形状

        # 加载所有 .mat 文件路径
        for path in paths:
            self.filenames += glob(f'{path}/**/user_*.mat', recursive=True)

        if not self.filenames:
            raise ValueError(f"{paths} not found data")
        
        print(f"find {len(self.filenames)} files")
        
        # 加载 embedding 信息
        if os.path.exists(embedding_path):
            self.embedding_data = np.load(embedding_path)['e']  # 加载字段 e
            if len(self.embedding_data) != len(self.filenames):
                print(f"Warning: The number of embeddings ({len(self.embedding_data)}) does not match the number of data files ({len(self.filenames)}).")
                # 处理数量不匹配的情况
                min_len = min(len(self.embedding_data), len(self.filenames))
                self.filenames = self.filenames[:min_len]
                self.embedding_data = self.embedding_data[:min_len]
        else:
            print(f"Warning: Embedding file {embedding_path} not found, creating random embeddings")
            # 创建随机嵌入作为备选
            self.embedding_data = np.random.randn(len(self.filenames), 128)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        
        try:
            # 加载当前数据文件
            cur_sample = scio.loadmat(cur_filename, verify_compressed_data_integrity=False)
        
            if 'feature' not in cur_sample:
                raise KeyError(f"data {cur_filename} loss 'feature'")
            
            # 加载数据并转换为实数
            cur_data = torch.from_numpy(cur_sample['feature']).float()  # 加载特征数据并转换为浮点数

            # 加载对应的 embedding 数据
            cur_cond = torch.from_numpy(self.embedding_data[idx]).float().squeeze(0)  # 加载并转换为浮点数

            # 调整数据的大小
            cur_data = self._resize_data(cur_data, self.target_shape)

            return {
                'data': cur_data,
                'cond': cur_cond
            }
        
        except Exception as e:
            print(f"Error loading file {cur_filename}: {e}")
            raise e

    def _resize_data(self, data, target_shape):
        """根据目标形状调整数据的大小（填充或裁剪）"""
        current_shape = data.shape
        
        if current_shape[0] < target_shape[0]:
            # 数据较小，填充到目标形状
            padding = (0, 0, 0, target_shape[0] - current_shape[0])  # 填充到目标行数
            data = torch.nn.functional.pad(data, padding, "constant", 0)
        elif current_shape[0] > target_shape[0]:
            # 数据较大，裁剪到目标形状
            data = data[:target_shape[0], :, :]

        # 确保列数符合目标形状（如果需要的话）
        if current_shape[1] != target_shape[1]:
            data = data[:, :target_shape[1], :]
        
        return data


class FMCWDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        try:
            cur_sample = scio.loadmat(cur_filename)
            cur_data = torch.from_numpy(cur_sample['feature']).to(torch.complex64)
            cur_cond = torch.from_numpy(cur_sample['cond'].astype(np.int16)).to(torch.complex64)
            return {
                'data': cur_data,
                'cond': cur_cond.squeeze(0)
            }
        except Exception as e:
            print(f"Error loading file {cur_filename}: {e}")
            raise e


class MIMODataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        dataset = scio.loadmat(self.filenames[idx])
        data = torch.from_numpy(dataset['down_link']).to(torch.complex64)
        cond = torch.from_numpy(dataset['up_link']).to(torch.complex64)
        return {
            'data': torch.view_as_real(data),
            'cond': torch.view_as_real(cond)
        }


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        paths = paths[0]
        self.filenames = []
        self.filenames += glob(f'{paths}/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        dataset = scio.loadmat(self.filenames[idx])
        data = torch.from_numpy(dataset['clean']).to(torch.complex64)
        cond = torch.from_numpy(dataset['disturb']).to(torch.complex64)
        return {
            'data': data,
            'cond': cond
        }


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        sample_rate = self.params.sample_rate
        task_id = self.params.task_id

        # 过滤掉无效的记录
        valid_batch = [record for record in minibatch if 'data' in record and 'cond' in record]
        
        if len(valid_batch) == 0:
            raise ValueError("No valid records in batch!")

        # 处理 WiFi Case
        if task_id == 0:
            processed_batch = []
            for record in valid_batch:
                if len(record['data']) < sample_rate:
                    continue  # 跳过长度不足的记录
                record['data'] = record['data'].unsqueeze(0)
                data = record['data'].permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / (down_sample.std() + 1e-8)
                record['data'] = norm_data.permute(2, 0, 1)
                processed_batch.append(record)
            
            if len(processed_batch) == 0:
                raise ValueError("No valid records after processing!")
                
            data = torch.stack([record['data'] for record in processed_batch])
            cond = torch.stack([record['cond'] for record in processed_batch])
            return {
                'data': data,
                'cond': cond,
            }

        # 处理 FMCW Case
        elif task_id == 1:
            processed_batch = []
            for record in valid_batch:
                if len(record['data']) < sample_rate:
                    continue
                data = torch.view_as_real(record['data']).permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / (down_sample.std() + 1e-8)
                record['data'] = norm_data.permute(2, 0, 1)
                processed_batch.append(record)
                
            if len(processed_batch) == 0:
                raise ValueError("No valid records after processing!")
                
            data = torch.stack([record['data'] for record in processed_batch])
            cond = torch.stack([record['cond'] for record in processed_batch])
            return {
                'data': data,
                'cond': torch.view_as_real(cond),
            }

        # 处理 MIMO Case
        elif task_id == 2:
            processed_batch = []
            for record in valid_batch:
                data = record['data']
                cond = record['cond']
                cond_std = cond.std() + 1e-8
                norm_data = (data) / cond_std
                norm_cond = (cond) / cond_std
                record['data'] = norm_data.reshape(14, 96, 26, 2).transpose(1, 2)
                record['cond'] = norm_cond.reshape(14, 96, 26, 2).transpose(1, 2)
                processed_batch.append(record)
                
            data = torch.stack([record['data'] for record in processed_batch])
            cond = torch.stack([record['cond'] for record in processed_batch])
            return {
                'data': data,
                'cond': cond,
            }

        # 处理 EEG Case
        elif task_id == 3:
            processed_batch = []
            for record in valid_batch:
                data = record['data']
                cond = record['cond']
                cond_std = cond.std() + 1e-8
                norm_data = data / cond_std
                norm_cond = cond / cond_std
                record['data'] = norm_data.reshape(512, 1, 1)
                record['cond'] = norm_cond.reshape(512)
                processed_batch.append(record)
                
            data = torch.stack([record['data'] for record in processed_batch])
            cond = torch.stack([record['cond'] for record in processed_batch])
            return {
                'data': torch.view_as_real(data),
                'cond': torch.view_as_real(cond),
            }
        
        # 处理流量预测 Case - 这里应该不会被调用，因为有专门的TrafficCollator
        elif task_id == 4:
            # 备用处理逻辑
            data = torch.stack([record['target_traffic'] for record in valid_batch])
            cond = torch.stack([record['hist_traffic'] for record in valid_batch])
            return {
                'data': data.permute(0, 2, 1).unsqueeze(-1),  # (B, n_apps, pred_len, 1)
                'cond': cond.permute(0, 2, 1)  # (B, n_apps, hist_len)
            }

        else:
            raise ValueError("Unexpected task_id.")


def from_path(params, is_distributed=False):
    """统一的数据集创建函数"""
    data_dir = params.data_dir
    task_id = params.task_id
    
    print(f"🔧 Creating dataset for task_id: {task_id}")
    
    if task_id == 4:  # 流量预测任务
        print("📊 Creating traffic prediction dataset with USER-BASED split...")
        
        try:
            if hasattr(params, 'data_dir') and params.data_dir:
                data_dir = Path(params.data_dir[0]) if isinstance(params.data_dir, list) else Path(params.data_dir)
            else:
                data_dir = Path("dataset/traffic")
            
            traffic_path = data_dir / getattr(params, 'traffic_path', 'traffic_data_new.npz')
            embedding_path = getattr(params, 'embedding_path', 'environment_embeddings.npz')
            
            print(f"🔍 Data paths:")
            print(f"  • Traffic: {traffic_path}")
            print(f"  • Embedding: {embedding_path}")
            
            # 🔧 重要：确保使用按用户划分的数据集
            print("🚨 使用按用户划分的数据集（USER-BASED SPLIT）")
            
            # 直接创建训练数据集，而不是数据加载器
            train_dataset = TrafficPredictionDataset(
                traffic_path=str(traffic_path),
                embedding_path=str(embedding_path),
                mode='train',
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                hist_len=getattr(params, 'input_seq_len', 148),
                pred_len=getattr(params, 'pred_len', 20),
                stride=1
            )
            
            # 创建数据整理器
            collator = TrafficCollator(normalize=True, noise_std=0.01)
            
            # 创建数据加载器
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=params.batch_size,
                collate_fn=collator.collate,
                shuffle=not is_distributed,
                num_workers=0,
                sampler=DistributedSampler(train_dataset) if is_distributed else None,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False
            )
            
            print(f"✅ USER-BASED traffic dataset created successfully!")
            print(f"📊 Training dataset info:")
            print(f"  • Users: {len(train_dataset.user_ids)}")
            print(f"  • Samples: {len(train_dataset)}")
            print(f"  • Batches: {len(dataloader)}")
            
            return dataloader
            
        except Exception as e:
            print(f"❌ Error creating traffic dataset: {e}")
            raise e
    
    # 其他任务的处理
    elif task_id == 0:
        dataset = WiFiDataset(data_dir)
    elif task_id == 1:
        dataset = FMCWDataset(data_dir)
    elif task_id == 2:
        dataset = MIMODataset(data_dir)
    elif task_id == 3:
        dataset = EEGDataset(data_dir)
    else:
        raise ValueError("Unexpected task_id.")
    
    # 创建标准数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=0,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )
    
    return dataloader


def from_path_inference(params):
    """推理时的数据集创建函数"""
    cond_dir = params.cond_dir
    task_id = params.task_id
    
    if task_id == 0:
        dataset = WiFiDataset(cond_dir)
    elif task_id == 1:
        dataset = FMCWDataset(cond_dir)
    elif task_id == 2:
        dataset = MIMODataset(cond_dir)
    elif task_id == 3:
        dataset = EEGDataset(cond_dir)
    elif task_id == 4:
        # 流量预测推理
        dataset = TrafficPredictionDataset(
            traffic_path=params.traffic_path,
            embedding_path=params.embedding_path,
            mode='test'
        )
    else:
        raise ValueError("Unexpected task_id.")
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=Collator(params).collate if task_id != 4 else TrafficCollator().collate,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )


# 辅助函数
def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


# 测试函数
def test_traffic_dataset():
    """测试流量数据集"""
    print("🧪 Testing TrafficDataset...")
    
    # 创建数据加载器
    try:
        train_loader, val_loader, test_loader = create_traffic_dataloaders_by_user(
            batch_size=4, hist_len=12, pred_len=6, num_workers=0
        )
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        return None, None, None
    
    # 测试数据加载
    print(f"\n🔍 Testing data loading...")
    try:
        batch = next(iter(train_loader))
        
        print(f"✅ Batch loaded successfully!")
        print(f"📊 Batch info:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  • {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  • {key}: {type(value)}")
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"❌ Error loading batch: {e}")
        return None, None, None


if __name__ == "__main__":
    # 运行测试
    train_loader, val_loader, test_loader = test_traffic_dataset()
    if train_loader is not None:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
