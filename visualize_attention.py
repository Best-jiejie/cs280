import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from gpt import GPT, GPTConfig

def visualize_attention():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 初始化一个小模型，这里强制关闭 Flash Attention，因为我们需要拿到 Attention 权重矩阵
    config = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=4, n_head=4, n_kv_head=2, n_embd=128, use_flash_attention=False)
    model = GPT(config).to(device)

    # 尝试加载检查点，如果没有则使用随机初始化的权重
    ckpt_path = "logs/checkpoint_micro.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            # Handle potential shape mismatches if checkpoint is from an older architecture
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint due to architecture mismatch: {e}")
            print("⚠️ Using randomly initialized weights for demonstration.")
    else:
        print("⚠️ No checkpoint found. Using randomly initialized weights for demonstration.")

    model.eval()

    prompt = "Once upon a time, there was a brave"
    print(f"Prompt: '{prompt}'")
    
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    tokens = [tokenizer.decode([i]) for i in idx[0].tolist()]

    with torch.no_grad():
        # 调用时要求返回 attention 权重
        logits, _, _, all_att_weights = model(idx, return_attention=True)

    # all_att_weights 是一个列表，包含了所有层的 attention 权重
    # 形状: [Layer_idx][Batch, n_head, SeqLen, SeqLen]
    
    if not all_att_weights:
        print("Failed to get attention weights. Make sure use_flash_attention=False.")
        return

    # 选择最后一层的 attention 权重进行可视化
    last_layer_att = all_att_weights[-1][0].cpu() # 取 Batch 0
    n_head = last_layer_att.shape[0]

    fig, axes = plt.subplots(1, n_head, figsize=(5 * n_head, 5))
    if n_head == 1:
        axes = [axes]
    
    fig.suptitle('Attention Weights (Last Layer)', fontsize=16)

    for h in range(n_head):
        ax = axes[h]
        im = ax.imshow(last_layer_att[h].numpy(), cmap='viridis')
        if h == n_head - 1:
            fig.colorbar(im, ax=ax)
        
        ax.set_title(f"Head {h + 1}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens, rotation=0)

    plt.tight_layout()
    
    os.makedirs("images", exist_ok=True)
    save_path = "images/attention_heatmap.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Attention heatmap saved to: {save_path}")

if __name__ == "__main__":
    visualize_attention()
