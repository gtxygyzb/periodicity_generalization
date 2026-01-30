#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined visualization script (modified per user request)
- Individual per-folder heatmaps preserved.
- Combined heatmaps: 1 x N subplots (left-to-right), shared colorbar.
- Grouped bar chart: x-axis = model names; each model has 3 bars (Extrapolation/Hollow/In-Distribution).
- Combined loss curves: same-color per model, solid lines, alpha distinguishes train/test.
"""
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.colors import PowerNorm
from collections import defaultdict

# -------------------------------------------------
# 自动寻找最大 epoch 的 train / test jsonl
# -------------------------------------------------
def find_latest_jsonl_files(work_dir):
    """
    在目录中查找 train_results_epoch*.jsonl 和 test_results_epoch*.jsonl
    返回最大 epoch 对应的 (train_file, test_file, max_epoch)
    """
    train_pattern = re.compile(r"train_results_epoch(\d+)\.jsonl")
    test_pattern = re.compile(r"test_results_epoch(\d+)\.jsonl")

    train_files = {}
    test_files = {}

    for fname in os.listdir(work_dir):
        train_match = train_pattern.match(fname)
        test_match = test_pattern.match(fname)

        if train_match:
            epoch = int(train_match.group(1))
            train_files[epoch] = os.path.join(work_dir, fname)

        if test_match:
            epoch = int(test_match.group(1))
            test_files[epoch] = os.path.join(work_dir, fname)

    common_epochs = sorted(set(train_files.keys()) & set(test_files.keys()))
    if not common_epochs:
        # 如果找不到训练/测试成对文件，仍然尝试至少找到 test 文件
        if test_files:
            max_epoch = max(test_files.keys())
            return train_files.get(max_epoch, None), test_files[max_epoch], max_epoch
        raise RuntimeError(f"No matching train/test jsonl files found in {work_dir}.")

    max_epoch = max(common_epochs)
    return train_files[max_epoch], test_files[max_epoch], max_epoch


# -------------------------------------------------
# 加载 JSONL（适配单个文件或多个文件列表）
# -------------------------------------------------
def load_jsonl_files(file_list):
    all_data = []
    for file_path in file_list:
        if file_path is None:
            continue
        loaded = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
                    loaded += 1
        print(f"Loaded {loaded} records from {file_path}")
    print(f"Total records loaded: {len(all_data)}")
    return all_data


# -------------------------------------------------
# 分类与统计（按三类）
# -------------------------------------------------
def classify_pair(p1, p2, border_pairs, hollow_pairs):
    pair = (p1, p2)
    if pair in border_pairs:
        return "Extrapolation"
    elif pair in hollow_pairs:
        return "Hollow"
    else:
        return "In-Distribution"

def analyze_test_by_category(data, strict_match=False):
    """
    仅统计测试集，并按 Extrapolation / Hollow / In-Distribution 分类，
    返回三类的平均准确率字典
    """
    PERIOD_RANGE = list(range(2, 17))

    BORDER_PAIRS = {
        *((p, q) for p in {2, 3} for q in PERIOD_RANGE),
        *((p, q) for p in {15, 16} for q in PERIOD_RANGE),
        *((p, q) for q in {2, 3} for p in PERIOD_RANGE),
        *((p, q) for q in {15, 16} for p in PERIOD_RANGE),
    }

    HOLLOW_SET = {8, 9, 10, 11}
    HOLLOW_PAIRS = {(p1, p2) for p1 in HOLLOW_SET for p2 in HOLLOW_SET}

    stats = {
        "Extrapolation": [],
        "Hollow": [],
        "In-Distribution": [],
    }

    for item in data:
        input_str = item.get("input", "")
        input_str = input_str.rstrip("=")
        nums = input_str.split("+")
        if len(nums) != 2:
            continue

        p1 = len(nums[0].strip())
        p2 = len(nums[1].strip())

        target = item.get("target", "")
        pred = item.get("pred", "")
        if len(target) != len(pred) or len(target) == 0:
            continue

        if strict_match:
            acc = 1.0 if target == pred else 0.0
        else:
            acc = sum(t == p for t, p in zip(target, pred)) / len(target)

        category = classify_pair(p1, p2, BORDER_PAIRS, HOLLOW_PAIRS)
        stats[category].append(acc)

    avg_acc = {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in stats.items()}
    return avg_acc


# -------------------------------------------------
# 单目录热力图绘制（保留）以及合并热力图函数（新增）
# -------------------------------------------------
def analyze_jsonl_data(data, strict_match=False):
    accuracy_dict = defaultdict(list)

    for item in data:
        input_str = item.get("input", "")
        if input_str.endswith("="):
            input_str = input_str[:-1]

        numbers = input_str.split("+")
        if len(numbers) != 2:
            continue

        len1 = len(numbers[0].strip())
        len2 = len(numbers[1].strip())

        target = item.get("target", "")
        pred = item.get("pred", "")
        if len(target) != len(pred) or len(target) == 0:
            continue

        if strict_match:
            accuracy = 1.0 if target == pred else 0.0
        else:
            correct_bits = sum(t == p for t, p in zip(target, pred))
            accuracy = correct_bits / len(target)

        accuracy_dict[(len1, len2)].append(accuracy)

    accuracy_avg = {k: (float(np.mean(v)) if v else 0.0) for k, v in accuracy_dict.items()}
    return accuracy_dict, accuracy_avg

def create_heatmap(accuracy_avg, title, save_path):
    size_range = range(2, 17)
    heatmap_data = np.zeros((len(size_range), len(size_range)))

    for i, len1 in enumerate(size_range):
        for j, len2 in enumerate(size_range):
            heatmap_data[i, j] = accuracy_avg.get((len1, len2), 0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data,
        xticklabels=size_range,
        yticklabels=size_range,
        annot=True,
        fmt=".2f",
        norm=PowerNorm(gamma=0.5),
        vmin=heatmap_data.min(),
        vmax=heatmap_data.max(),
        square=True,
        cbar_kws={"label": "Accuracy"},
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Length of Second Number")
    plt.ylabel("Length of First Number")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved heatmap to {save_path}")

def plot_combined_heatmaps(
    model_acc_avg_dict,
    save_path,
    size_range=None,
):
    """
    把多张「单独 heatmap」横向放在一张图里：
    - 每个子图独立 cmap / vmin / vmax / colorbar
    - 不共享 norm
    - 不用 tight_layout
    """

    if not model_acc_avg_dict:
        raise ValueError("No heatmap data provided.")

    if size_range is None:
        size_range = list(range(2, 17))

    models = list(model_acc_avg_dict.keys())
    n_models = len(models)

    # ---------- 创建 figure ----------
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(5.2 * n_models, 4.8),
        constrained_layout=True,
    )

    if n_models == 1:
        axes = [axes]

    # ---------- 每个模型单独画一张 heatmap ----------
    for ax, model_name in zip(axes, models):
        accuracy_avg = model_acc_avg_dict[model_name]

        heatmap_data = np.zeros((len(size_range), len(size_range)))
        for i, len1 in enumerate(size_range):
            for j, len2 in enumerate(size_range):
                heatmap_data[i, j] = accuracy_avg.get((len1, len2), 0.0)

        sns.heatmap(
            heatmap_data,
            ax=ax,
            xticklabels=size_range,
            yticklabels=size_range,
            square=True,
            annot=False,                 # 和你现在一致
            cbar=True,                   # ★ 每张图自己的 colorbar
            cbar_kws={"label": "Accuracy"},
            vmin=0.0,                    # 和 create_heatmap 一致
            vmax=1.0,
        )

        ax.set_title(model_name, fontsize=14)
        ax.set_xlabel("Length of Second Number")
        ax.set_ylabel("Length of First Number")

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined independent heatmaps to {save_path}")



# -------------------------------------------------
# 解析 loss.txt（返回按 epoch 排序的 epochs, train_losses, test_losses）
# -------------------------------------------------
def parse_loss_file(loss_file):
    if not os.path.exists(loss_file):
        return None

    epoch_map = {}
    pattern = re.compile(r"Epoch\s*(\d+)\s*,\s*(Train|Test)\s*Loss\s*([0-9eE\+\-\.]+)")
    with open(loss_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                kind = m.group(2)
                loss = float(m.group(3))
                if epoch not in epoch_map:
                    epoch_map[epoch] = {}
                epoch_map[epoch][kind] = loss

    if not epoch_map:
        return None

    epochs_sorted = sorted(epoch_map.keys())
    train_losses = [epoch_map[e].get("Train", np.nan) for e in epochs_sorted]
    test_losses = [epoch_map[e].get("Test", np.nan) for e in epochs_sorted]

    return {
        "epochs": epochs_sorted,
        "train_losses": train_losses,
        "test_losses": test_losses
    }


# -------------------------------------------------
# 合并绘制：按模型为横坐标，每个模型并列三条柱子的函数（新增）
# models_acc_dict: { model_name: {"Extrapolation": v, "Hollow": v, "In-Distribution": v} }
# -------------------------------------------------
def plot_models_grouped_bar(models_acc_dict, save_path):
    models = list(models_acc_dict.keys())
    categories = ["In-Distribution", "Hollow", "Extrapolation"]
    n_models = len(models)
    n_cats = len(categories)

    x = np.arange(n_models)
    total_width = 0.75
    bar_width = total_width / n_cats

    plt.figure(figsize=(max(6, 1.4 * n_models), 3.2))  # 关键：压高度
    cmap = plt.get_cmap("tab10")

    for j, cat in enumerate(categories):
        vals = [models_acc_dict[m].get(cat, 0.0) for m in models]
        offsets = x - total_width / 2 + j * bar_width + bar_width / 2
        bars = plt.bar(
            offsets, vals,
            width=bar_width,
            label=cat,
            color=cmap(j)
        )
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha='center',
                va='bottom',
                fontsize=7
            )

    plt.xticks(x, models, fontsize=9)
    plt.ylabel("Accuracy", fontsize=9)
    plt.ylim(0.0, 1.05)

    # ---------- 关键：legend 顶部横排 ----------
    plt.legend(
        ncol=3,                # 一行
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        frameon=True,
        fontsize=8
    )
    # -----------------------------------------

    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()



# -------------------------------------------------
# 合并绘制 loss 曲线（保持不变）
# -------------------------------------------------
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

def plot_combined_loss_curves(loss_info_dict, save_path):
    # -------- 关键：只调字号，不改字体 --------
    plt.rcParams.update({
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    plt.figure(figsize=(6, 4))
    cmap = plt.get_cmap("tab10")
    models = list(loss_info_dict.keys())

    model_lines = []
    model_labels = []

    for idx, model in enumerate(models):
        info = loss_info_dict[model]
        if info is None:
            continue

        epochs = info["epochs"]
        train_losses = info["train_losses"]
        test_losses = info["test_losses"]

        color = cmap(idx % 10)

        plt.plot(epochs, train_losses, color=color, linewidth=2.0, alpha=1.0)
        plt.plot(epochs, test_losses,  color=color, linewidth=2.0, alpha=0.45)

        model_lines.append(Line2D([0], [0], color=color, linewidth=2.0))
        model_labels.append(model)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("In-distribution vs Out-of-distribution Test Set Loss Curves")
    plt.grid(True)

    # -------- Model legend：左下 --------
    legend_models = plt.legend(
        model_lines,
        model_labels,
        loc="upper left",
        frameon=True,
        framealpha=0.85
    )

    # -------- Train/Test legend：左上 --------
    proxy_train = Line2D([0], [0], color="black", linewidth=2.0, alpha=1.0)
    proxy_test  = Line2D([0], [0], color="black", linewidth=2.0, alpha=0.45)

    legend_phase = plt.legend(
        [proxy_train, proxy_test],
        ["In-Distribution (ID)", "Out-of-Distribution (OOD)"],
        loc="lower left",
        frameon=True,
        framealpha=0.85
    )

    plt.gca().add_artist(legend_models)

    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------------------------------
# 从文件夹名中提取模型名（尽可能稳健）
# -------------------------------------------------
def extract_model_name_from_dir(dir_path):
    base = os.path.basename(dir_path.rstrip("/\\"))
    
    # 模式1：匹配 _add_{模型名}_ 这种特定模式
    # 寻找 seq_add_ 或 add_ 后跟模型名，然后跟着 _日期_时间
    m = re.search(r'(?:seq_)?add_([A-Za-z0-9_.]+?)_\d{2}-\d{2}_\d{2}-\d{2}', base)
    if m:
        model_name = m.group(1)
        # 如果模型名中包含点号，可能需要进一步处理
        if "." in model_name:
            # 例如 Qwen2.5Embedding-Transformer -> 提取 Transformer
            if "-" in model_name:
                return model_name.split("-")[-1]
            # 或者保留点号后的部分
            return model_name.split(".")[-1]
        return model_name
    
    # 模式2：更通用的匹配，但避免匹配日期部分
    # 匹配 -模型名_ 但确保后面不是日期模式
    m = re.search(r'-([A-Za-z0-9_.]+?)_(?!\d{2}-\d{2})', base)
    if m:
        return m.group(1)
    
    # 模式3：匹配最后一个非日期部分
    # 分割下划线，找到不是日期格式的部分
    parts = base.split("_")
    for i in range(len(parts)-1, -1, -1):
        part = parts[i]
        # 跳过日期格式的部分（如 01-08, 12-16）
        if re.match(r'\d{2}-\d{2}', part):
            continue
        # 检查是否是模型名（包含字母）
        if re.search(r'[A-Za-z]', part):
            # 检查是否有连字符分隔的复合模型名
            if "-" in part and re.search(r'[A-Za-z]', part.split("-")[-1]):
                return part.split("-")[-1]
            return part
    
    # 备用方案
    return base


# -------------------------------------------------
# 主流程（接受多个文件夹）
# -------------------------------------------------
def main(work_dirs, combined_out):
    print("Combined analysis for directories:")
    for d in work_dirs:
        print(" -", d)

    # 用于收集跨目录数据
    models_acc = {}           # model_name -> {"Extrapolation": v, ...}
    models_loss_info = {}     # model_name -> parse_loss_file result
    models_heatmap_avg = {}   # model_name -> accuracy_avg (用于合并热力图)

    # 创建输出目录
    # combined_out = os.path.join(".", "combined_results")
    
    os.makedirs(combined_out, exist_ok=True)

    for work_dir in work_dirs:
        if not os.path.isdir(work_dir):
            print(f"Warning: {work_dir} is not a directory, skipping.")
            continue

        model_name = extract_model_name_from_dir(work_dir)

        try:
            train_file, test_file, max_epoch = find_latest_jsonl_files(work_dir)
        except RuntimeError as e:
            print(f"Could not find jsonl files in {work_dir}: {e}")
            continue

        print(f"\nDirectory: {work_dir}")
        print(f"Model: {model_name}")
        print(f"Using epoch {max_epoch}")
        print(f"Train file: {train_file}")
        print(f"Test file:  {test_file}")

        # 读取测试文件并做分析
        data_test = load_jsonl_files([test_file])
        _, accuracy_avg = analyze_jsonl_data(data_test, strict_match=False)

        # 保留并保存单目录 heatmap（原有行为）
        heatmap_path = os.path.join(work_dir, f"accuracy_heatmap_epoch{max_epoch}.png")
        try:
            create_heatmap(
                accuracy_avg,
                title=f"Per-Bit Accuracy Heatmap (Epoch {max_epoch}) - {model_name}",
                save_path=heatmap_path,
            )
        except Exception as e:
            print(f"[WARN] failed to create per-folder heatmap for {work_dir}: {e}")

        # 收集数据用于合并图
        models_heatmap_avg[model_name] = accuracy_avg
        cat_acc = analyze_test_by_category(data_test, strict_match=False)
        models_acc[model_name] = cat_acc

        # 解析 loss.txt（可能不存在）
        loss_file = os.path.join(work_dir, "loss.txt")
        loss_info = parse_loss_file(loss_file)
        if loss_info is None:
            print(f"loss.txt not found or empty in {work_dir}, skipping loss for this model.")
        models_loss_info[model_name] = loss_info

    # 1) 合并热力图（所有模型放在一张图的子图中，左到右排列）
    if models_heatmap_avg:
        combined_heatmap_path = os.path.join(combined_out, "combined_accuracy_heatmaps.png")
        try:
            plot_combined_heatmaps(models_heatmap_avg, combined_heatmap_path)
        except Exception as e:
            print(f"[WARN] failed to create combined heatmaps: {e}")
    else:
        print("No heatmap data collected; skipping combined heatmaps.")

    # 2) 合并柱状图（横轴为模型，每个模型有三条并列柱）
    if models_acc:
        combined_bar_path = os.path.join(combined_out, "combined_test_category_accuracy.png")
        try:
            plot_models_grouped_bar(models_acc, combined_bar_path)
        except Exception as e:
            print(f"[WARN] failed to create grouped bar chart: {e}")
    else:
        print("No models' category accuracy collected; skipping combined grouped bar chart.")

    # 3) 合并 loss 曲线（所有模型一起）
    if any(v is not None for v in models_loss_info.values()):
        combined_loss_path = os.path.join(combined_out, "combined_loss_curve.png")
        try:
            plot_combined_loss_curves(models_loss_info, combined_loss_path)
        except Exception as e:
            print(f"[WARN] failed to create combined loss curves: {e}")
    else:
        print("No loss info found for any model; skipping combined loss curves.")


if __name__ == "__main__":
    work_dirs = [
        "./RQ/RQ4/3_2seq_conv_Qwen2.5Embedding-Transformer_01-08_09-28",
        "./RQ/RQ4/5_2seq_conv_Qwen2.5Embedding-FANformer_01-08_09-29",
        "./RQ/RQ4/3_2seq_conv_Qwen2.5Embedding-KAN_01-08_09-28",
        "./RQ/RQ4/2seq_conv_Mamba_01-05_23-08",
        "./RQ/RQ4/2seq_conv_RWKV_01-07_15-23",
    ]
    combined_out = "./RQ/RQ4/combined_results"
    main(work_dirs, combined_out)
