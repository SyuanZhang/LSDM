# eval.py
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader

# 你工程里的模块
from params import all_params
from wifi_model import tfdiff_WiFi
from diffusion import SignalDiffusion, GaussianDiffusion
from dataset import from_path  # 用于触发数据模块（也便于 fallback）
try:
    # 很可能存在这个类（根据你日志里 “Initializing TrafficDataset ...”）
    from dataset import TrafficDataset
except Exception:
    TrafficDataset = None

# 评估工具（你 util.py 已经有）
from util import (
    seed_everything, setup_tf32, get_device, dataloader_kwargs,
    evaluate_forecast, summarize_eval
)

def build_params(task_id: int, device_str: str):
    """和你的 train.py 一致：task_id=4 时手工拼一份默认参数。"""
    if task_id == 4:
        from params import AttrDict
        import numpy as np
        return AttrDict(
            task_id=4,
            log_dir='./log/traffic_eval',
            model_dir='./model/traffic',
            data_dir=['./dataset/traffic'],
            traffic_path='traffic_data_new.npz',
            embedding_path='environment_embeddings.npz',
            max_iter=10000,
            batch_size=64,
            learning_rate=1e-4,
            max_grad_norm=0.5,
            inference_batch_size=64,
            robust_sampling=True,
            seq_len=24,
            pred_len=20,
            sample_rate=20,
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
            learn_tfdiff=False,
            max_step=1000,
            signal_diffusion=True,
            blur_schedule=((1e-5**2) * np.ones(1000)).tolist(),
            noise_schedule=np.linspace(1e-4, 0.02, 1000).tolist(),
            device=device_str
        )
    return all_params[task_id]

def build_loader(params, split: str, batch_size: int, num_workers: int):
    if TrafficDataset is not None:
        # --- 只构造 traffic_path ---
        def _abs_path(p):
            if os.path.isabs(p):
                return p
            root = params.data_dir[0] if isinstance(params.data_dir, (list, tuple)) else params.data_dir
            return os.path.abspath(os.path.join(root, p))

        traffic_path = _abs_path(params.traffic_path)
        
        if not os.path.exists(traffic_path):
            raise FileNotFoundError(f"Traffic data file not found: {traffic_path}")

        # --- 只传递必需参数 ---
        dataset = TrafficDataset(
            mode=split,
            traffic_path=traffic_path,
            # 不传 embedding_path
        )

        loader = DataLoader(
            dataset,
            **dataloader_kwargs(
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=False,
            )
        )
        return loader

    print("[WARN] dataset.TrafficDataset 不可用，退化为 from_path(params) 的默认 DataLoader")
    loader = from_path(params)
    return loader


def load_model_and_diffusion(params, device):
    model = tfdiff_WiFi(params).to(device)
    diffusion = (SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)).to(device)
    return model, diffusion

def load_checkpoint_into_model(model, ckpt_path: str, map_location):
    if not os.path.exists(ckpt_path):
        # 尝试 {model_dir}/weights.pt
        alt = os.path.join(os.path.abspath(os.path.dirname(ckpt_path)), 'weights.pt')
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)

def run_eval(args):
    seed_everything(42, deterministic=False)
    setup_tf32(True)

    device = get_device(args.device)
    params = build_params(args.task_id, device.type)

    # DataLoader
    loader = build_loader(
        params,
        split=args.split,
        batch_size=args.batch_size or params.inference_batch_size,
        num_workers=args.num_workers
    )

    # Model & diffusion
    model, diffusion = load_model_and_diffusion(params, device)
    model.eval()

    # Checkpoint
    ckpt = args.ckpt or os.path.join(params.model_dir, 'weights.pt')
    print(f"🔁 Loading checkpoint: {ckpt}")
    load_checkpoint_into_model(model, ckpt, map_location=device)
    print("✅ Checkpoint loaded.")

    # 推理并累积所有 batch（和训练一致：退化→还原）
    y_preds, y_trues = [], []
    stats_for_eval = None

    n_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for i, features in enumerate(loader, 1):
            # 移到设备；dataset 默认已经是张量字典
            for k, v in list(features.items()):
                if torch.is_tensor(v):
                    features[k] = v.to(device, non_blocking=True)

            # 取真值（推荐用 target_traffic；如果没有就用 data.squeeze(-1)）
            if 'target_traffic' in features:
                y_true = features['target_traffic']              # [B, A, T]
            else:
                data = features['data']                           # [B, A, T, 1]
                y_true = data.squeeze(-1)

            # 退化时间步
            B = y_true.shape[0]
            t = torch.randint(0, diffusion.max_step, [B], dtype=torch.int64, device=device)

            # 退化输入（训练里是对 data 做的）
            x0 = features['data'] if 'data' in features else y_true.unsqueeze(-1)
            x_t = diffusion.degrade_fn(x0, t, params.task_id)

            # 模型还原
            y_pred = model(x_t, t, features['cond'])
            if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)

            y_preds.append(y_pred.detach().cpu())
            y_trues.append(y_true.detach().cpu())

            if stats_for_eval is None and 'stats' in features:
                stats_for_eval = features['stats']  # dict: {'mean':..., 'std':...}

            if args.limit and sum(p.size(0) for p in y_preds) >= args.limit:
                break

            if i % 20 == 0 or i == n_batches:
                done = sum(p.size(0) for p in y_preds)
                print(f"[{i}/{n_batches}] accumulated {done} samples...")

    import numpy as np
    y_pred_all = torch.cat(y_preds, dim=0).numpy()  # [N, A, T]
    y_true_all = torch.cat(y_trues, dim=0).numpy()  # [N, A, T]

    # 评估
    res = evaluate_forecast(
        y_pred_all, y_true_all,
        stats=stats_for_eval,
        layout='BAT',
        per_app=True,
        per_horizon=True
    )

    # 打印摘要
    print("\n================ Evaluation Summary ================")
    print(summarize_eval(res))
    print("====================================================")
    print(f"耗时: {time.time() - t0:.1f}s  样本数: {y_true_all.shape[0]}")

    # 保存结果
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"eval_{args.split}.json")
    import json as _json
    with open(out_json, "w", encoding="utf-8") as f:
        _json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"📄 Saved metrics -> {out_json}")

def main():
    p = argparse.ArgumentParser("Evaluate tfdiff model")
    p.add_argument("--task_id", type=int, default=4, help="0/1/2/3/4 -> WiFi/FMCW/MIMO/EEG/Traffic")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint 路径（默认使用 {model_dir}/weights.pt）")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"], help="评估数据划分")
    p.add_argument("--batch_size", type=int, default=None, help="评估 batch size（默认用 params.inference_batch_size）")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers（Windows 推荐 0~2）")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="评估设备")
    p.add_argument("--limit", type=int, default=None, help="仅评估前 N 条样本（调试用）")
    p.add_argument("--out_dir", type=str, default="./eval_out", help="指标保存目录")
    args = p.parse_args()
    run_eval(args)

if __name__ == "__main__":
    main()
