import os
import json
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from gpt import GPT, GPTConfig
from transformers import AutoTokenizer

# ==========================================
# 新增功能：读取日志并绘制对比曲线图
# ==========================================
def plot_and_save_metrics():
    print("📊 正在生成数据对比折线图...")
    os.makedirs("images", exist_ok=True) # 创建专门保存图片的文件夹

    models = ["micro", "mini", "small"]
    colors = {"micro": "blue", "mini": "orange", "small": "green"}
    labels = {"micro": "Micro GPT (~6.5M)", "mini": "Mini GPT (~23M)", "small": "Small GPT (~34M)"}

    data = {}
    for m in models:
        log_path = f"logs/training_log_{m}.json"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                data[m] = json.load(f)
        else:
            print(f"⚠️ 找不到 {m} 的日志文件，将无法在图表中显示。")
            data[m] = []

    # 画图 1：Training Loss vs Iteration
    plt.figure(figsize=(10, 6))
    for m in models:
        if data[m]:
            iters = [entry["iteration"] for entry in data[m]]
            train_losses = [entry["train_loss"] for entry in data[m]]
            plt.plot(iters, train_losses, label=labels[m], color=colors[m], linewidth=2)

    plt.title("Training Loss across Different Model Sizes")
    plt.xlabel("Iterations")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    train_loss_path = "images/train_loss_comparison.png"
    plt.savefig(train_loss_path, dpi=300, bbox_inches='tight') # 高清保存
    plt.close()

    # 画图 2：Validation Perplexity vs Iteration
    plt.figure(figsize=(10, 6))
    for m in models:
        if data[m]:
            iters = [entry["iteration"] for entry in data[m]]
            val_ppls = [entry["val_perplexity"] for entry in data[m]]
            plt.plot(iters, val_ppls, label=labels[m], color=colors[m], linewidth=2)

    plt.title("Validation Perplexity across Different Model Sizes")
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    # 限制 y 轴范围，防止初期的极端值把曲线压得太扁
    plt.ylim(0, 50)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    val_ppl_path = "images/val_perplexity_comparison.png"
    plt.savefig(val_ppl_path, dpi=300, bbox_inches='tight') # 高清保存
    plt.close()

    print(f"✅ 图片已成功保存至项目下的 'images/' 文件夹！\n")

# ==========================================
# 核心功能：带 Temperature, Top-K 和 Top-P 采样的生成
# ==========================================
@torch.no_grad()
def generate_text(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, *_ = model(idx_cond) # 防弹解包
        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ==========================================
# 主评估流程
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on {device}...\n")

    # 1. 优先执行画图并保存图片的功能
    plot_and_save_metrics()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_text = "Once upon a time, there was a brave little"
    print(f"=== Shared Prompt: '{prompt_text}' ===\n")
    prompt_idx = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    vocab_size = tokenizer.vocab_size
    model_configs = {
        "Micro": GPTConfig(vocab_size=vocab_size, n_layer=4, n_head=4, n_embd=128),
        "Mini": GPTConfig(vocab_size=vocab_size, n_layer=6, n_head=6, n_embd=384),
        "Small": GPTConfig(vocab_size=vocab_size, n_layer=8, n_head=8, n_embd=512)
    }

    num_samples = 3
    max_tokens = 50
    temperature = 0.8
    top_k = 40
    top_p = 0.9

    for model_name, config in model_configs.items():
        print(f"{'='*50}")
        print(f"Evaluating {model_name} GPT")
        print(f"{'='*50}")

        checkpoint_path = f"logs/checkpoint_{model_name.lower()}.pt"
        log_path = f"logs/training_log_{model_name.lower()}.json"

        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
                if logs:
                    final_val_ppl = logs[-1].get("val_perplexity", "N/A")
                    print(f"Quantitative -> Final Validation Perplexity: {final_val_ppl:.4f}\n")

        if not os.path.exists(checkpoint_path):
            print(f"⚠️ No checkpoint found for {model_name}. Skipping generation.\n")
            continue

        model = GPT(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Qualitative -> Generating {num_samples} samples (Temp={temperature}, Top-K={top_k}, Top-P={top_p}):")

        for i in range(num_samples):
            torch.manual_seed(1337 + i) # 固定种子，保证每次生成的样本多样但可复现
            out_idx = generate_text(model, prompt_idx, max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            generated_text = tokenizer.decode(out_idx[0].tolist())
            print(f"\n[Sample {i+1}]: {generated_text}")
        print("\n")

if __name__ == "__main__":
    main()