import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm
from collections import defaultdict

# -------------------------------------------------
# 工具函数 1：自动寻找最大 epoch 的 train / test jsonl
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
    仅统计测试集，并按 Extrapolation / Hollow / In-Distribution 分类
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
        input_str = item["input"].rstrip("=")
        nums = input_str.split("+")
        if len(nums) != 2:
            continue

        p1 = len(nums[0].strip())
        p2 = len(nums[1].strip())

        target = item["target"]
        pred = item["pred"]
        if len(target) != len(pred):
            continue

        if strict_match:
            acc = 1.0 if target == pred else 0.0
        else:
            acc = sum(t == p for t, p in zip(target, pred)) / len(target)

        category = classify_pair(p1, p2, BORDER_PAIRS, HOLLOW_PAIRS)
        stats[category].append(acc)

    # 计算平均正确率
    avg_acc = {
        k: np.mean(v) if len(v) > 0 else 0.0
        for k, v in stats.items()
    }

    return avg_acc

def plot_test_category_bar(avg_acc, save_path, model_name="FANformer",):
    """
    avg_acc: dict with keys {"Extrapolation", "Hollow", "In-Distribution"}
    model_name: str, e.g., "FANformer"
    """

    categories = ["Extrapolation", "Hollow", "In-Distribution"]
    values = [avg_acc[c] for c in categories]

    x = np.array([0])  # single model, reserved for future extension
    bar_width = 0.22

    plt.figure(figsize=(6, 4))

    bars = []
    bars.append(
        plt.bar(x - bar_width, values[0], width=bar_width, label="Extrapolation")
    )
    bars.append(
        plt.bar(x, values[1], width=bar_width, label="Hollow")
    )
    bars.append(
        plt.bar(x + bar_width, values[2], width=bar_width, label="In-Distribution")
    )

    # 数值标注
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.xticks(x, [model_name])
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy by Data Category")
    plt.legend(frameon=False)

    # 自动 y 轴范围，留一点 headroom
    #ymin = min(values) - 0.05
    #ymax = max(values) + 0.05
    #plt.ylim(max(0, ymin), min(1.0, ymax))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved test category bar chart to {save_path}")


def find_latest_jsonl_files(work_dir):
    """
    在目录中查找 train_results_epoch*.jsonl 和 test_results_epoch*.jsonl
    返回最大 epoch 对应的 (train_file, test_file)
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
        raise RuntimeError("No matching train/test jsonl files found.")

    max_epoch = max(common_epochs)
    return train_files[max_epoch], test_files[max_epoch], max_epoch


# -------------------------------------------------
# 工具函数 2：加载 JSONL
# -------------------------------------------------
def load_jsonl_files(file_list):
    all_data = []
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
        print(f"Loaded {len(all_data)} records from {file_path}")
    print(f"Total records loaded: {len(all_data)}")
    return all_data


# -------------------------------------------------
# 工具函数 3：准确率分析
# -------------------------------------------------
def analyze_jsonl_data(data, strict_match=False):
    accuracy_dict = defaultdict(list)

    for item in data:
        input_str = item["input"]
        if input_str.endswith("="):
            input_str = input_str[:-1]

        numbers = input_str.split("+")
        if len(numbers) != 2:
            continue

        len1 = len(numbers[0].strip())
        len2 = len(numbers[1].strip())

        target = item["target"]
        pred = item["pred"]
        if len(target) != len(pred):
            continue

        if strict_match:
            accuracy = 1.0 if target == pred else 0.0
        else:
            correct_bits = sum(t == p for t, p in zip(target, pred))
            accuracy = correct_bits / len(target)

        accuracy_dict[(len1, len2)].append(accuracy)

    accuracy_avg = {
        k: np.mean(v) if v else 0.0
        for k, v in accuracy_dict.items()
    }
    return accuracy_dict, accuracy_avg


# -------------------------------------------------
# 工具函数 4：绘制热力图
# -------------------------------------------------
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


# -------------------------------------------------
# 工具函数 5：解析并绘制 loss 曲线
# -------------------------------------------------
def plot_loss_curve(loss_file, save_path):
    epochs = []
    train_losses = []
    test_losses = []

    pattern = re.compile(r"Epoch (\d+), (Train|Test) Loss ([\d\.eE+-]+)")

    with open(loss_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                kind = match.group(2)
                loss = float(match.group(3))

                if kind == "Train":
                    epochs.append(epoch)
                    train_losses.append(loss)
                else:
                    test_losses.append(loss)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Testing Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved loss curve to {save_path}")


# -------------------------------------------------
# 主流程
# -------------------------------------------------
def main(work_dir):
    print(f"Working directory: {work_dir}")

    train_file, test_file, max_epoch = find_latest_jsonl_files(work_dir)
    print(f"Using epoch {max_epoch}")
    print(f"Train file: {train_file}")
    print(f"Test file:  {test_file}")

    # data = load_jsonl_files([train_file, test_file])
    data = load_jsonl_files([test_file])

    # Heatmap
    _, accuracy_avg = analyze_jsonl_data(data, strict_match=False)
    heatmap_path = os.path.join(work_dir, f"accuracy_heatmap_epoch{max_epoch}.png")
    create_heatmap(
        accuracy_avg,
        title=f"Per-Bit Accuracy Heatmap (Epoch {max_epoch})",
        save_path=heatmap_path,
    )

    # Loss curve
    loss_file = os.path.join(work_dir, "loss.txt")
    if os.path.exists(loss_file):
        loss_curve_path = os.path.join(work_dir, "loss_curve.png")
        plot_loss_curve(loss_file, loss_curve_path)
    else:
        print("loss.txt not found, skipping loss curve.")

    # -------------------------
    # 测试集三类柱状图
    # -------------------------
    test_acc_by_category = analyze_test_by_category(
        data, strict_match=False
    )

    bar_path = os.path.join(work_dir, "test_category_accuracy.png")
    plot_test_category_bar(test_acc_by_category, bar_path)



# -------------------------------------------------
# 入口
# -------------------------------------------------
if __name__ == "__main__":
    work_dir = "./final_res/2seq_add_Qwen2.5Embedding-FANformer_12-17_03-16"
    main(work_dir)
    
    work_dir = "./final_res/2seq_add_Qwen2.5Embedding-Transformer_12-17_03-16"
    main(work_dir)

    
