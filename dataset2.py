import os
import json
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        """
        Custom PyTorch Dataset for pre-training GPT.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length + 1,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0).long()

        # 自回归：x 是输入，y 是错开一位的预测目标
        x = input_ids[:-1].clone()
        y = input_ids[1:].clone()

        return x, y

def prepare_dataloaders(max_samples=50000, val_ratio=0.05, max_length=256, batch_size=16):
    """
    加载数据，优先从本地 data 文件夹读取，如果不存在则从 Hugging Face 下载并保存到本地。
    """
    # 确保 data 文件夹存在
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # 定义缓存文件的路径，文件名包含样本数量以便区分不同的子集
    data_file = os.path.join(data_dir, f"tinystories_subset_{max_samples}.json")

    # ==========================================
    # 核心修改：本地缓存逻辑
    # ==========================================
    if os.path.exists(data_file):
        print(f"Loading dataset from local cache: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
    else:
        print("Downloading TinyStories dataset from Hugging Face...")
        ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        texts = list(ds['text'])

        print(f"Saving dataset to local cache: {data_file}")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False)
    # ==========================================

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    total_size = len(texts)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    full_dataset = TinyStoriesDataset(texts, tokenizer, max_length)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, getattr(tokenizer, "vocab_size", len(tokenizer))

if __name__ == "__main__":
    # 使用 10000 或50000个样本进行本地测试
    train_loader, val_loader, vocab_size = prepare_dataloaders(max_samples=50000)
    print(f"Vocabulary Size: {vocab_size}")