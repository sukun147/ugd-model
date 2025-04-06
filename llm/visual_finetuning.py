import json
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
# 1. 加载和预处理日志数据
# ------------------------------
def load_logs(log_path):
    """读取 JSONL 文件并返回平滑处理后的 DataFrame"""
    logs = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 跳过无效行

    df = pd.DataFrame(logs)

    # 平滑处理 Loss（窗口=10）
    if "loss" in df.columns:
        df["smoothed_loss"] = df["loss"].rolling(window=10, min_periods=1).mean()

    return df


# ------------------------------
# 2. 绘制并保存平滑后的 Loss 曲线
# ------------------------------
def plot_smoothed_loss(df, save_path="data/loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(
        df["current_steps"],
        df["smoothed_loss"],
        color="darkred",
        linewidth=2,
        label="Loss"
    )
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"平滑 Loss 曲线已保存至: {save_path}")
    plt.close()


# ------------------------------
# 3. 绘制并保存学习率曲线
# ------------------------------
def plot_lr(df, save_path="data/lr_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(df["current_steps"], df["lr"], color="blue", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.yscale("log")  # 对数坐标
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"LR 曲线已保存至: {save_path}")
    plt.close()


# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    # 参数配置
    LOG_PATH = "../data/trainer_log.jsonl"  # 日志文件路径

    # 加载数据并平滑处理
    df = load_logs(LOG_PATH)

    # 检查必要字段
    required_columns = ["current_steps", "loss", "lr"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("日志文件缺少必要字段！确保包含: current_steps, loss, lr")

    # 绘制并保存图片
    plot_smoothed_loss(df)
    plot_lr(df)
