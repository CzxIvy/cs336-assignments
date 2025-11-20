import os
import sys
import json
import torch
import pathlib
import numpy as np
from tqdm import tqdm
from model import BasicsTransformerLM
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from tests.adapters import *

data_dir = pathlib.Path(__file__).resolve().parent / 'data'
train_data_path = os.path.join(data_dir, 'train.dat')
valid_data_path = os.path.join(data_dir, 'valid.dat')
config_path = pathlib.Path(__file__).resolve().parent / 'config.json'
save_dir = pathlib.Path(__file__).resolve().parent / 'checkpoints'

device = 'cuda'

def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode='r')
    return arr

def get_batch(memmap_arr, batch_size, context_length):
    start_idxs = np.random.randint(
        low=0,
        high=len(memmap_arr)-context_length,
        size=(batch_size,)
    )
    inputs = np.stack([memmap_arr[start_idx:start_idx+context_length] for start_idx in start_idxs])
    labels = np.stack([memmap_arr[start_idx+1:start_idx+context_length+1] for start_idx in start_idxs])
    return torch.from_numpy(inputs).to(device).long(), torch.from_numpy(labels).to(device).long()

def memmap_val_iterator(memmap_arr, batch_size, context_length):
    start_num = (len(memmap_arr)-context_length-1) // batch_size
    for i in range(start_num):
        start = i*batch_size
        inputs = np.stack([memmap_arr[i:i+context_length] for i in range(start, start+batch_size)])
        labels = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(start, start+batch_size)])
        yield torch.from_numpy(inputs).to(device).long(), torch.from_numpy(labels).to(device).long()
        
if __name__ == '__main__':
    # 1. 导入数据集和配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    model = BasicsTransformerLM(**config['model'])
    
    params = {}
    for group in config.values():
        params.update(group)
    
    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    
    args = DotDict(params)
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 加载数据集
    train_data = get_memmap_dataset(train_data_path)
    valid_data = get_memmap_dataset(valid_data_path)
    
    # 3. 构建优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 4. 恢复断点
    start_iter = 0
    if args.resume_checkpoint != 0:
        print(f'Resuming from checkpoint {args.resume_checkpoint}')
        resume_ckpt_path = os.path.join(save_dir, f'ckpt_iter{args.resume_checkpoint}.pt')
        start_iter = run_load_checkpoint(src=resume_ckpt_path, model=model, optimizer=optimizer)
        print(f'Resumed at iteration {start_iter}')
    
    # 5. 训练loop
    for iter in tqdm(range(start_iter, args.train_steps), desc='Training'):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length)
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = run_cross_entropy(
            inputs=logits.reshape(-1, logits.shape[-1]),
            targets=y.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(parameters=list(model.parameters()), max_l2_norm=args.clip_grad_norm)
        
        lr = run_get_lr_cosine_schedule(
            it=iter,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if (iter+1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses_list = []
                count = 0
                for x_val, y_val in memmap_val_iterator(valid_data, args.batch_size, args.context_length):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    val_losses = run_cross_entropy(
                        inputs=val_logits.reshape(-1, logits.shape[-1]),
                        targets=y_val.reshape(-1)
                    )
                    val_losses_list.append(val_losses.item())
                    count += 1
                    if count >= args.val_batches:
                        break
                val_loss_mean = np.mean(val_losses_list)
                print(f'iter {iter:05d}: VALID loss = {val_loss_mean:.4f}')
                
        if (iter+1) % args.save_interval == 0:
            ckpt_name = os.path.join(save_dir, f'ckpt_iter{iter+1}.pt')
            run_save_checkpoint(model=model, optimizer=optimizer, iteration=iter, out=ckpt_name)
            print(f'Checkpoint saved to {ckpt_name}')