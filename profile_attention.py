import torch
import time
import matplotlib.pyplot as plt
import os
from gpt import GPTConfig, CausalSelfAttention

def profile_attention():
    os.makedirs("images", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Flash Attention requires CUDA. CPU is not supported for this benchmark.")
        return

    print("Benchmarking Naive vs Flash Attention...")
    
    # 模拟一个稍微大一点的上下文，以显现差异
    config = GPTConfig(block_size=1024, n_embd=768, n_head=12, n_kv_head=12)
    
    batch_size = 8
    seq_len = 1024
    
    naive_attn = CausalSelfAttention(config).to(device)
    naive_attn.use_flash_attention = False # Force naive

    flash_attn = CausalSelfAttention(config).to(device)
    flash_attn.use_flash_attention = True # Force flash
    
    x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

    results = {"Naive": {}, "Flash": {}}

    for name, model in [("Naive", naive_attn), ("Flash", flash_attn)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        for _ in range(5):
            _ = model(x, position_ids)
            
        torch.cuda.synchronize()
        start = time.time()
        
        # Benchmark Time
        iters = 50
        for _ in range(iters):
            _ = model(x, position_ids)
            
        torch.cuda.synchronize()
        end = time.time()
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        avg_time = ((end - start) / iters) * 1000 # ms
        
        results[name]["memory"] = peak_mem
        results[name]["time"] = avg_time
        
        print(f"[{name} Attention] Peak Memory: {peak_mem:.2f} MB, Time per forward: {avg_time:.2f} ms")

    # 画图
    labels = list(results.keys())
    memories = [results[l]["memory"] for l in labels]
    times = [results[l]["time"] for l in labels]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    x_pos = [0, 1]
    width = 0.35

    bars1 = ax1.bar([p - width/2 for p in x_pos], memories, width, label='Peak Memory (MB)', color='royalblue')
    ax1.set_ylabel('Peak GPU Memory (MB)', color='royalblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar([p + width/2 for p in x_pos], times, width, label='Forward Time (ms)', color='darkorange')
    ax2.set_ylabel('Forward Time (ms)', color='darkorange', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=12)
    plt.title('Naive vs Flash Attention (Batch=8, Seq=1024)', fontsize=14, fontweight='bold')

    for rect in bars1:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for rect in bars2:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    save_path = "images/flash_attention_benchmark.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Profiling plot saved to: {save_path}")

if __name__ == "__main__":
    profile_attention()
