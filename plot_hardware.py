import matplotlib.pyplot as plt
import numpy as np
import os

def plot_hardware_metrics():
    os.makedirs("images", exist_ok=True)

    # ==========================================
    # 📝 请在这里填入你实际观察到的（或估算的）数据
    # ==========================================
    models = ['Micro (~6.5M)', 'Mini (~23M)', 'Small (~34M)']

    # 填入三个模型的显存峰值占用 (单位：GB)
    # (比如你截图里 Small 大约占了 5.2GB)
    memory_usage = [1.2, 3.5, 5.2]

    # 填入三个模型跑 2000 步大约花费的时间 (单位：分钟)
    # (如果是 Small 跑 10000 步，你可以按比例折算成 2000 步的时间，保持横向对比的公平性)
    training_time = [15, 35, 60]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 画第一组柱子：显存占用 (蓝色)
    bars1 = ax1.bar(x - width/2, memory_usage, width, label='Peak Memory (GB)', color='#1f77b4', alpha=0.8)
    ax1.set_ylabel('Peak GPU Memory (GB)', color='#1f77b4', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0, max(memory_usage) + 2)

    # 创建共用 x 轴的第二个 y 轴
    ax2 = ax1.twinx()

    # 画第二组柱子：训练时间 (橙色)
    bars2 = ax2.bar(x + width/2, training_time, width, label='Training Time (mins / 2000 steps)', color='#ff7f0e', alpha=0.8)
    ax2.set_ylabel('Training Time (Minutes)', color='#ff7f0e', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, max(training_time) + 20)

    # 标签和标题
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    plt.title('Hardware Efficiency Comparison: Memory vs. Training Time', fontsize=14, fontweight='bold', pad=20)

    # 添加柱子上的具体数值标签
    def autolabel(rects, ax, suffix=''):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}{suffix}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(bars1, ax1, 'GB')
    autolabel(bars2, ax2, 'm')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    save_path = "images/hardware_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 硬件对比柱状图已高清保存至: {save_path}")

if __name__ == "__main__":
    plot_hardware_metrics()