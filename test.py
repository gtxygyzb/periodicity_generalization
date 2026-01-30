import torch
import math
import json

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from generate_periodic_data import gen_periodic_data, plot_periodic_data

import argparse

def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证 CUDA 可复现（非常重要）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 固定随机种子
SEED = 42
set_seed(SEED)

model_names = ['FAN', 'FANGated', 'MLP', 'KAN', 'Transformer', 'FANformer', 
               'Qwen2.5Embedding-FANformer', 'OnlyNormNet_params',
               'Qwen2.5Embedding-Transformer','Mamba', 'RWKV']
periodic_types = ['2seq_add', '2seq_div', 'seq', 'seq2', 'seq3', 'seq4', 'sin', 'mod', 'complex_1', 'complex_2', 'complex_3', 'complex_4', 'complex_5', 'complex_6']

parser = argparse.ArgumentParser()
parser.add_argument('--periodic_type', type=str, choices=periodic_types, help='periodic type', default='sin')
parser.add_argument('--path', type=str, help='path')
parser.add_argument('--model_name', type=str, choices=model_names, help='model name', default='FAN')
parser.add_argument('--dynamic_tanh', action='store_true', help='use dynamic tanh')
parser.add_argument('--sin', action='store_true', help='use sin')
parser.add_argument('--alpha_init_value', type=float, default=0.5, help='initial value of alpha in dynamic tanh and sin')
parser.add_argument('--layers', type=int, default=3, help='number of layers')

args = parser.parse_args()
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')

def plot_layernorm_io(module, input, output, layer_id=0, save_dir=".", epoch=None):
    if epoch is None:
        return  

    x_in = input[0].detach().cpu().numpy().flatten()
    x_out = output.detach().cpu().numpy().flatten()

    plt.figure(figsize=(6, 5))
    # 绘制输入和输出的直方图
    sns.histplot(x_in, bins=80, color='skyblue', label='Before LayerNorm', stat='density', kde=True, alpha=0.4)
    sns.histplot(x_out, bins=80, color='orange', label='After LayerNorm', stat='density', kde=True, alpha=0.4)

    # 添加均值和方差信息
    plt.axvline(x_in.mean(), color='blue', linestyle='--', lw=1, label=f'Before mean={x_in.mean():.2f}, std={x_in.std():.2f}')
    plt.axvline(x_out.mean(), color='red', linestyle='--', lw=1, label=f'After mean={x_out.mean():.2f}, std={x_out.std():.2f}')

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"LayerNorm distribution (layer{layer_id}, epoch={epoch})")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"attn_norm_hist_layer{layer_id}_epoch{epoch}.png")
    plt.savefig(fname, dpi=300)
    plt.close()



t, data, t_test, data_test, PERIOD, BATCHSIZE, NUMEPOCH, PRINTEPOCH, lr, wd, y_uper, y_lower = gen_periodic_data(args.periodic_type)

    
import os
path = args.path

if args.sin:
    path = f'{path}_sin_alpha{args.alpha_init_value}'
elif args.dynamic_tanh:
    path = f'{path}_dynamic_tanh_alpha{args.alpha_init_value}'

if not os.path.exists(f'{path}'):
    os.makedirs(f'{path}')


from torch.utils.data import Dataset, DataLoader

VOCAB = {
    "+": 10,
    "=": 28,
    # "-": 12,
    # ".": 13,
    "0": 15, "1": 16, "2": 17, "3": 18, "4": 19,
    "5": 20, "6": 21, "7": 22, "8": 23, "9": 24
}
def tokenize_number_str(num_str: str):
    """字符串 -> token id 列表"""
    return [VOCAB[ch] for ch in num_str if ch in VOCAB]

class NumberStringDataset(Dataset):
    def __init__(self, xs, ys, periods=10):
        self.xs = xs  # str
        self.ys = ys  # str
        self.periods = periods # turple

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.periods[idx]

# 创建 dataset
if args.periodic_type == '2seq_add' or args.periodic_type == '2seq_div':
    dataset_train = NumberStringDataset(t, data, y_uper)
    dataset_test = NumberStringDataset(t_test, data_test, y_lower)
else:
    dataset_train = NumberStringDataset(t, data)
    dataset_test = NumberStringDataset(t_test, data_test)

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=BATCHSIZE, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=BATCHSIZE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
from architecture import get_model_by_name
# from dynamic_tanh import convert_ln_to_dyt

print(f'model name: {args.model_name}')

output_dim = len(VOCAB)
model = get_model_by_name(args.model_name, output_dim=output_dim, num_layers=args.layers, num_heads=8).to(device)

print(model)
# 这里用一个可变 dict 存 epoch
hook_context = {"epoch": None}

def hook_fn(m, inp, out, layer_id):
    if hook_context["epoch"] is not None:
        plot_layernorm_io(m, inp, out, layer_id=layer_id, save_dir=path, epoch=hook_context["epoch"])


# 给第 1、2、3 层 attn_norm 注册 hook
#for lid in [0, 1, 2]:
#   norm_layer = model.layers[lid]
#   norm_layer.register_forward_hook(lambda m, inp, out, lid=lid: hook_fn(m, inp, out, lid))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

# train
num_epochs = NUMEPOCH
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/home/gtxygyzb/models/Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0  # 也可以是你 vocab 里没有的数字
CHUNK_LEN =16 # For RWKV

ID2LOCAL = {qid: i for i, qid in enumerate(VOCAB.values())}
LOCAL2ID = {i: qid for i, qid in enumerate(VOCAB.values())}

def pad_to_chunk(x, chunk_len=CHUNK_LEN, pad_value=PAD_ID): 
    """
    For RWKV
    将 tensor 的时间维度 pad 到 chunk_len 的倍数
    x: [B, T, ...] 或 [B, T]
    """
    B, T = x.shape[:2]
    T_pad = math.ceil(T / chunk_len) * chunk_len
    if T_pad == T:
        return x
    pad_size = T_pad - T
    pad_shape = (B, pad_size) + x.shape[2:]
    pad_tensor = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=1)

loss_file = f'{path}/loss.txt'

if os.path.exists(loss_file):
    os.remove(loss_file)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    results_train = []  # 用于保存训练样本前 1000 条
    for x_batch, y_batch, p_batch in tqdm(dataloader_train):
        input_ids = [torch.tensor(tokenize_number_str(x), dtype=torch.long) for x in x_batch]
        output_ids = [torch.tensor(tokenize_number_str(y), dtype=torch.long) for y in y_batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID).to(device)
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=PAD_ID).to(device)

        if args.model_name == 'RWKV':
            input_ids = pad_to_chunk(input_ids, chunk_len=CHUNK_LEN, pad_value=PAD_ID)
            output_ids = pad_to_chunk(output_ids, chunk_len=CHUNK_LEN, pad_value=PAD_ID)

        output_ids_local = output_ids.clone()
        for qid, lid in ID2LOCAL.items():
            output_ids_local[output_ids == qid] = lid
        optimizer.zero_grad()
        logits = model(input_ids)   # [batch, seq_len, |VOCAB|]

        # ------------------ mask: 前一个周期不算 loss ------------------
        mask = torch.zeros_like(output_ids_local, dtype=torch.bool)

        for i, p in enumerate(p_batch):
            period = p_batch[0][i]
            cut_point = p_batch[1][i]
            mask[i, cut_point-1:period-1] = True # 从第三个周期cutpoint到第三个周期结束算loss
            #print(i, ":", cut_point-1, 4*period)
        mask = mask & (output_ids != PAD_ID)

        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = output_ids_local.reshape(-1)
        mask_flat = mask.reshape(-1)
        

        loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred_ids = logits.argmax(dim=-1)

        ID2TOKEN = {i: tok for i, tok in enumerate(VOCAB.keys())}
        for i, seq in enumerate(pred_ids.cpu()):
            if len(results_train) >= 1000:
                break
            tokens = [ID2TOKEN[i.item()] for i in seq]
            number_text = "".join(tokens)
            period = p_batch[0][i]
            cut_point = p_batch[1][i]
            
            results_train.append({
                "input": x_batch[i][:cut_point],
                "target": y_batch[i][cut_point-1:period-1],
                "pred": number_text[cut_point-1:period-1]
            })

    if epoch % PRINTEPOCH == 0:
        print(f'Epoch {epoch}, Train Loss {total_loss / len(dataloader_train)}')
        with open(loss_file, 'a') as f:
            f.write(f'Epoch {epoch}, Train Loss {total_loss / len(dataloader_train)}\n')
        
        result_file = f"{path}/train_results_epoch{epoch}.jsonl"
        with open(result_file, "w", encoding="utf-8") as f:
            for r in results_train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        model.eval()
        result = []
        total_test_loss = 0
        with torch.no_grad():
            hook_context["epoch"] = epoch
            for x_batch, y_batch, p_batch in tqdm(dataloader_test):
                input_ids = [torch.tensor(tokenize_number_str(x), dtype=torch.long) for x in x_batch]
                output_ids = [torch.tensor(tokenize_number_str(y), dtype=torch.long) for y in y_batch]

                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID).to(device)
                output_ids = pad_sequence(output_ids, batch_first=True, padding_value=PAD_ID).to(device)

                if args.model_name == 'RWKV':
                    input_ids = pad_to_chunk(input_ids, chunk_len=CHUNK_LEN, pad_value=PAD_ID)
                    output_ids = pad_to_chunk(output_ids, chunk_len=CHUNK_LEN, pad_value=PAD_ID)

                output_ids_local = output_ids.clone()
                for qid, lid in ID2LOCAL.items():
                    output_ids_local[output_ids == qid] = lid
                optimizer.zero_grad()
                logits = model(input_ids)   # [batch, seq_len, |VOCAB|]
                pred_ids = logits.argmax(dim=-1)

                ID2TOKEN = {i: tok for i, tok in enumerate(VOCAB.keys())}
                for i, seq in enumerate(pred_ids.cpu()):
                    tokens = [ID2TOKEN[i.item()] for i in seq]
                    number_text = "".join(tokens)
                    period = p_batch[0][i]
                    cut_point = p_batch[1][i]

                    result.append({
                        "input": x_batch[i][:cut_point],
                        "target": y_batch[i][cut_point-1:period-1],
                        "pred": number_text[cut_point-1:period-1]
                    })

                mask = torch.zeros_like(output_ids_local, dtype=torch.bool)
                for i, p in enumerate(p_batch):
                    period = p_batch[0][i]
                    cut_point = p_batch[1][i]
                    mask[i, cut_point-1:period-1] = True # 从第三个周期cutpoint到第三个周期结束算loss
                mask = mask & (output_ids != PAD_ID)

                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = output_ids_local.reshape(-1)
                mask_flat = mask.reshape(-1)
                test_loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])

                total_test_loss += test_loss.item()
            hook_context["epoch"] = None  # 重置

        print(f'Epoch {epoch}, Test Loss {total_test_loss / len(dataloader_test)}')
        with open(loss_file, 'a') as f:
            f.write(f'Epoch {epoch}, Test Loss {total_test_loss / len(dataloader_test)}\n')
        # 保存预测结果（推荐jsonl格式）
        result_file = f"{path}/test_results_epoch{epoch}.jsonl"
        with open(result_file, "w", encoding="utf-8") as f:
            for r in result:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # plot
        # plot_periodic_data(t, data, t_test, data_test, result, args, epoch, path, y_uper, y_lower)

model_save_dir = "./models"
if args.sin:
    torch.save(model.state_dict(), f'{model_save_dir}/{args.model_name}_{args.periodic_type}_sin.pth')
elif args.dynamic_tanh:
    torch.save(model.state_dict(), f'{model_save_dir}/{args.model_name}_{args.periodic_type}_dynamic_tanh.pth')
else:
    torch.save(model.state_dict(), f'{model_save_dir}/{args.model_name}_{args.periodic_type}_.pth')
#torch.save(model.state_dict(), f'{args.model_name}/.pth')

# model.eval()
# 
# total_test_loss = 0
# with torch.no_grad():
#     for x_batch, y_batch in tqdm(dataloader_test):
#         input_ids = [tokenize_number_str(x) for x in x_batch]
#         input_ids = torch.tensor(input_ids).to(device)
# 
#         output_ids = [tokenize_number_str(y) for y in y_batch]
#         output_ids = torch.tensor(output_ids).to(device)
# 
#         output_ids_local = output_ids.clone()
#         for qid, lid in ID2LOCAL.items():
#             output_ids_local[output_ids == qid] = lid
#         optimizer.zero_grad()
#         logits = model(input_ids)   # [batch, seq_len, |VOCAB|]
# 
#         mask = torch.zeros_like(output_ids_local, dtype=torch.bool)
#         mask[:, period-1:] = True  # 从第一个周期结束开始算 loss
#         logits_flat = logits.reshape(-1, logits.size(-1))
#         labels_flat = output_ids_local.reshape(-1)
#         mask_flat = mask.reshape(-1)
#         test_loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])
#         total_test_loss += test_loss.item()
# 
#     print(f'Final Epoch, Test Loss {total_test_loss / len(dataloader_test)}')
#     with open(loss_file, 'a') as f:
#         f.write(f'Final Epoch, Test Loss {total_test_loss / len(dataloader_test)}\n')
