import os
import time
import torch
import matplotlib.pyplot as plt
from gpt import GPT, GPTConfig
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 文本生成函数：支持开启或关闭 KV Cache
# ==========================================
@torch.no_grad()
def generate_text_speed_test(model, tokenizer, prompt, max_new_tokens, use_cache=False):
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # GPU 测速需要同步，确保计时准确
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    past_kv = None
    current_idx = idx

    for i in range(max_new_tokens):
        # 1. 核心区别：是否开启缓存
        if use_cache:
            if past_kv is not None:
                # 如果有缓存，模型只需要看最新生成的那个词（1个 token）
                current_idx = idx_next
            else:
                # 第一次运行，必须看完整的 prompt
                current_idx = idx

            logits, _, past_kv = model(current_idx, use_cache=True, past_key_values=past_kv)
        else:
            # 如果没有缓存，模型每次都要重新看前面所有的词
            idx_cond = idx[:, -model.config.block_size:]
            logits, _, _ = model(idx_cond, use_cache=False)

        # 2. 获取最后一个 token 的预测结果
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 3. 拼接到总序列中
        idx = torch.cat((idx, idx_next), dim=1)

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    tokens_per_sec = max_new_tokens / total_time
    generated_text = tokenizer.decode(idx[0].tolist())

    return generated_text, total_time, tokens_per_sec

# ==========================================
# 画图函数：生成报告所需的柱状图
# ==========================================
def plot_speed_comparison(speed_no_cache, speed_with_cache):
    os.makedirs("images", exist_ok=True)
    labels = ['Without KV Cache', 'With KV Cache']
    speeds = [speed_no_cache, speed_with_cache]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, speeds, color=['#3498db', '#2ecc71'], width=0.5)

    plt.title('Inference Speed Comparison (Advanced Task B)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Generation Speed (Tokens / Second)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上显示具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f} tok/s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    save_path = "images/kv_cache_speed_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 测速对比柱状图已保存至: {save_path}")

# ==========================================
# 主流程
# ==========================================
def main():
    print(f"🚀 Running Advanced Task (KV Cache) Evaluation on {device}...")

    # 1. 加载 Tokenizer 和 测试 Prompt
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_text = "Once upon a time, there was a brave little dog named Max. Max loved to explore the forest. One day, he found a mysterious cave."
    max_new_tokens = 200  # 生成长一点的文本更能拉开缓存的差距

    # 2. 只使用最大的模型 (Small GPT) 进行高级任务评估
    model_name = "Small"
    checkpoint_path = f"logs/checkpoint_small.pt"

    if not os.path.exists(checkpoint_path):
        print("❌ 找不到 Small GPT 的模型权重文件，请确认训练已完成。")
        return

    config = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config).to(device)

    print(f"📦 Loading {model_name} GPT checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 为了防止首次运行的 GPU 预热影响时间，先空跑一次
    print("🔥 Warming up GPU...")
    _ = generate_text_speed_test(model, tokenizer, "Test", 10, use_cache=False)

    print("\n" + "="*50)
    print(f"🧪 Test Prompt: '{prompt_text}'")
    print(f"🎯 Target Length: {max_new_tokens} new tokens")
    print("="*50)

    # 3. 测试【关闭】KV Cache
    print("\n⏳ [1/2] Generating WITHOUT KV Cache...")
    text_no_cache, time_no_cache, tps_no_cache = generate_text_speed_test(model, tokenizer, prompt_text, max_new_tokens, use_cache=False)
    print(f"   ⏱️ Time taken: {time_no_cache:.2f} seconds")
    print(f"   🚀 Speed: {tps_no_cache:.2f} tokens/sec")

    # 4. 测试【开启】KV Cache
    print("\n⚡ [2/2] Generating WITH KV Cache...")
    text_with_cache, time_with_cache, tps_with_cache = generate_text_speed_test(model, tokenizer, prompt_text, max_new_tokens, use_cache=True)
    print(f"   ⏱️ Time taken: {time_with_cache:.2f} seconds")
    print(f"   🚀 Speed: {tps_with_cache:.2f} tokens/sec")

    # 验证两次生成的文本是否一致（确保 KV cache 逻辑是对的）
    if text_no_cache == text_with_cache:
        print("\n✅ Verification Passed: The generated texts are exactly the same!")
    else:
        print("\n⚠️ Warning: The generated texts differ slightly. Check your temperature/sampling settings if you used them.")

    # 5. 画出漂亮的对比图用于报告
    plot_speed_comparison(tps_no_cache, tps_with_cache)

    speedup_ratio = tps_with_cache / tps_no_cache
    print(f"\n🎉 Conclusion: KV Cache accelerated generation by {speedup_ratio:.2f}x times!")

if __name__ == "__main__":
    main()