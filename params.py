import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

# ========================
# Wifi Parameter Setting.
# ========================
params_wifi = AttrDict(
    task_id=0,
    
    log_dir='./log/traffic',
    model_dir='./model/traffic/b16-128-1000s',
    data_dir=['./dataset/traffic'],
    out_dir='./dataset/traffic/output',
    cond_dir=['./dataset/traffic/cond'],
    fid_pred_dir='./dataset/traffic/img_matric/pred',
    fid_data_dir='./dataset/traffic/img_matric/data',
    
    # 数据路径
    traffic_path='traffic_data.npz',
    embedding_path='environment_embeddings.npz',
    
    # 训练参数
    max_iter=None,  # 通过命令行参数控制
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    
    # 推理参数
    inference_batch_size=8,
    robust_sampling=True,
    
    # 数据参数
    seq_len=24,      # 历史序列长度
    pred_len=12,     # 预测序列长度
    sample_rate=12,  # 采样率，与pred_len一致
    input_dim=20,    # 输入维度（应用数量）
    extra_dim=[128], # 额外维度（环境嵌入维度）
    cond_dim=276,    # 条件维度：20(流量) + 128(历史环境) + 128(预测环境)
    
    # 模型参数
    embed_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_block=8,
    dropout=0.1,
    mlp_ratio=4.0,
    learn_tfdiff=False,
    
    # 扩散参数
    signal_diffusion=True,
    max_step=1000,
    blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
    noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
    device='cuda'
)

# ========================
# FMCW Parameter Setting.
# ========================
params_fmcw = AttrDict(
    task_id=1,
    
    log_dir='./log/traffic',
    model_dir='./model/traffic/b16-128-1000s',
    data_dir=['./dataset/traffic'],
    out_dir='./dataset/traffic/output',
    cond_dir=['./dataset/traffic/cond'],
    fid_pred_dir='./dataset/traffic/img_matric/pred',
    fid_data_dir='./dataset/traffic/img_matric/data',
    
    # 数据路径
    traffic_path='traffic_data.npz',
    embedding_path='environment_embeddings.npz',
    
    # 训练参数
    max_iter=None,  # 通过命令行参数控制
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    
    # 推理参数
    inference_batch_size=8,
    robust_sampling=True,
    
    # 数据参数
    seq_len=24,      # 历史序列长度
    pred_len=12,     # 预测序列长度
    sample_rate=12,  # 采样率，与pred_len一致
    input_dim=20,    # 输入维度（应用数量）
    extra_dim=[128], # 额外维度（环境嵌入维度）
    cond_dim=276,    # 条件维度：20(流量) + 128(历史环境) + 128(预测环境)
    
    # 模型参数
    embed_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_block=8,
    dropout=0.1,
    mlp_ratio=4.0,
    learn_tfdiff=False,
    
    # 扩散参数
    signal_diffusion=True,
    max_step=1000,
    blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
    noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
    device='cuda'
)

# =======================
# MIMO Parameter Setting.
# =======================
params_mimo = AttrDict(
    task_id=2,
    
    log_dir='./log/traffic',
    model_dir='./model/traffic/b16-128-1000s',
    data_dir=['./dataset/traffic'],
    out_dir='./dataset/traffic/output',
    cond_dir=['./dataset/traffic/cond'],
    fid_pred_dir='./dataset/traffic/img_matric/pred',
    fid_data_dir='./dataset/traffic/img_matric/data',
    
    # 数据路径
    traffic_path='traffic_data.npz',
    embedding_path='environment_embeddings.npz',
    
    # 训练参数
    max_iter=None,  # 通过命令行参数控制
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    
    # 推理参数
    inference_batch_size=8,
    robust_sampling=True,
    
    # 数据参数
    seq_len=24,      # 历史序列长度
    pred_len=12,     # 预测序列长度
    sample_rate=12,  # 采样率，与pred_len一致
    input_dim=20,    # 输入维度（应用数量）
    extra_dim=[128], # 额外维度（环境嵌入维度）
    cond_dim=276,    # 条件维度：20(流量) + 128(历史环境) + 128(预测环境)
    
    # 模型参数
    embed_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_block=8,
    dropout=0.1,
    mlp_ratio=4.0,
    learn_tfdiff=False,
    
    # 扩散参数
    signal_diffusion=True,
    max_step=1000,
    blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
    noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
    device='cuda'
)


# ======================
# EEG Parameter Setting. 
# ======================
params_eeg = AttrDict(
    task_id=0,

    log_dir='./log/traffic',
    model_dir='./model/traffic/b16-128-1000s',
    data_dir=['./dataset/traffic'],
    out_dir='./dataset/traffic/output',
    cond_dir=['./dataset/traffic/cond'],
    fid_pred_dir='./dataset/traffic/img_matric/pred',
    fid_data_dir='./dataset/traffic/img_matric/data',
    
    # 数据路径
    traffic_path='traffic_data.npz',
    embedding_path='environment_embeddings.npz',
    
    # 训练参数
    max_iter=None,  # 通过命令行参数控制
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    
    # 推理参数
    inference_batch_size=8,
    robust_sampling=True,
    
    # 数据参数
    seq_len=24,      # 历史序列长度
    pred_len=12,     # 预测序列长度
    sample_rate=12,  # 采样率，与pred_len一致
    input_dim=20,    # 输入维度（应用数量）
    extra_dim=[128], # 额外维度（环境嵌入维度）
    cond_dim=276,    # 条件维度：20(流量) + 128(历史环境) + 128(预测环境)
    
    # 模型参数
    embed_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_block=8,
    dropout=0.1,
    mlp_ratio=4.0,
    learn_tfdiff=False,
    
    # 扩散参数
    signal_diffusion=True,
    max_step=1000,
    blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
    noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
    device='cuda'
)
# 在params.py中添加流量预测参数
# params.py 的最后部分修改为：

# 流量预测参数
# ======================
# 流量预测参数设置
# ======================
params_traffic = AttrDict(
    task_id=4,
    log_dir='./log/traffic',
    model_dir='./model/traffic/b16-128-1000s',
    data_dir=['./dataset/traffic'],
    out_dir='./dataset/traffic/output',
    cond_dir=['./dataset/traffic/cond'],
    fid_pred_dir='./dataset/traffic/img_matric/pred',
    fid_data_dir='./dataset/traffic/img_matric/data',
    
    # 数据路径
    traffic_path='traffic_data.npz',
    embedding_path='environment_embeddings.npz',
    
    # 训练参数
    max_iter=None,  # 通过命令行参数控制
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    
    # 推理参数
    inference_batch_size=8,
    robust_sampling=True,
    
    # 数据参数
    seq_len=24,      # 历史序列长度
    pred_len=16,     # 预测序列长度
    sample_rate=16,  # 采样率，与pred_len一致
    input_dim=20,    # 输入维度（应用数量）
    extra_dim=[128], # 额外维度（环境嵌入维度）
    cond_dim=276,    # 条件维度：20(流量) + 128(历史环境) + 128(预测环境)
    
    # 模型参数
    embed_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_block=8,
    dropout=0.1,
    mlp_ratio=4.0,
    learn_tfdiff=False,
    
    # 扩散参数
    signal_diffusion=True,
    max_step=1000,
    blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
    noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
    device='cuda'
)

# 包含所有任务的参数列表
all_params = [params_wifi, params_fmcw, params_mimo, params_eeg, params_traffic]
