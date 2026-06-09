import os
import math
import json
import torch
from dataset import prepare_dataloaders
from gpt import GPT, GPTConfig

# ==========================================
# 🏅 正式训练超参数设置 (Formal Training)
# ==========================================
batch_size = 16
max_samples = 50000
learning_rate = 3e-4     # 稍微调低一点学习率，让后期的 Loss 下降更平稳
max_iters = 10000        # 🎯 正式训练目标：10000步！
eval_interval = 500      # 每 500 步评估一次，节省时间
eval_iters = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 辅助函数：评估 Loss 和 Perplexity
# ==========================================
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    out = {}
    model.eval()
    loaders = {'train': train_loader, 'val': val_loader}

    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(loader):
            if k >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            # 防弹写法：吸收所有多余的返回值
            _, loss, *_ = model(X, targets=Y)
            losses[k] = loss.item()

        avg_loss = losses.mean().item()
        out[split] = avg_loss
        out[f'{split}_perplexity'] = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    model.train()
    return out

# ==========================================
# 辅助函数：生成样本文本
# ==========================================
@torch.no_grad()
def generate_sample(model, tokenizer, prompt="Once upon a time", max_new_tokens=50):
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, *_ = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return tokenizer.decode(idx[0].tolist())

# ==========================================
# 主训练流程
# ==========================================
def main():
    print(f"Using device: {device}")
    os.makedirs("logs", exist_ok=True)

    train_loader, val_loader, vocab_size = prepare_dataloaders(max_samples=max_samples, batch_size=batch_size)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 🎯 集中算力：注释掉 Micro 和 Mini，只跑 Small GPT
    model_configs = {
        "Micro": GPTConfig(vocab_size=vocab_size, n_layer=4, n_head=4, n_kv_head=2, n_embd=128, use_flash_attention=True),
        "Mini": GPTConfig(vocab_size=vocab_size, n_layer=6, n_head=6, n_kv_head=2, n_embd=384, use_flash_attention=True),
        "Small": GPTConfig(vocab_size=vocab_size, n_layer=8, n_head=8, n_kv_head=2, n_embd=512, use_flash_attention=True)
    }

    for model_name, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"🚀 STARTING FORMAL OVERNIGHT TRAINING FOR: {model_name} GPT")
        print(f"🎯 Target Iterations: {max_iters}")
        print(f"{'='*50}")

        checkpoint_path = f"logs/checkpoint_{model_name.lower()}.pt"
        log_path = f"logs/training_log_{model_name.lower()}.json"

        model = GPT(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

        start_iter = 0
        training_logs = []

        # 强大的断点续训：如果你明天早上起来发现只跑了 8000 步断电了，再运行一次它会自动从 8000 接着跑！
        if os.path.exists(checkpoint_path):
            print(f"Loading {model_name} checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iteration'] + 1

            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    training_logs = json.load(f)

        if start_iter >= max_iters:
            print(f"✅ {model_name} GPT is already fully trained to {max_iters} steps!")
            continue

        train_iter = iter(train_loader)

        for iter_num in range(start_iter, max_iters):
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                metrics = estimate_loss(model, train_loader, val_loader)
                sample_text = generate_sample(model, tokenizer)

                log_entry = {
                    "iteration": iter_num,
                    "train_loss": metrics['train'],
                    "val_loss": metrics['val'],
                    "val_perplexity": metrics['val_perplexity'],
                    "sample_text": sample_text
                }
                training_logs.append(log_entry)

                print(f"[{model_name}] Step {iter_num}/{max_iters}: Train Loss {metrics['train']:.4f}, Val PPL {metrics['val_perplexity']:.4f}")
                print(f"Sample: {sample_text}\n" + "-"*40)

                # 固化状态到硬盘
                torch.save({
                    'iteration': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(training_logs, f, indent=4, ensure_ascii=False)

            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)

            xb, yb = xb.to(device), yb.to(device)

            _, loss, *_ = model(xb, targets=yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"✅ Finished formal training {model_name} GPT! You can now write an amazing report!")

if __name__ == "__main__":
    main()