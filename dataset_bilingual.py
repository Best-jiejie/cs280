import os
import json
import torch
import urllib.request
import random
from torch.utils.data import Dataset, random_split, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class BilingualDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
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

        # Autoregressive shifting
        x = input_ids[:-1].clone()
        y = input_ids[1:].clone()

        return x, y

def get_hongloumeng_texts(max_length=200):
    """
    Downloads and chunks Dream of the Red Chamber.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    hlm_file = os.path.join(data_dir, "hongloumeng.txt")
    
    if not os.path.exists(hlm_file):
        print("Downloading Dream of the Red Chamber...")
        # A stable source for classic Chinese texts
        url = "https://raw.githubusercontent.com/shibing624/text_error_correction/master/pycorrector/data/hongloumeng.txt"
        try:
            # Add headers to avoid 403 Forbidden
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(hlm_file, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Download failed: {e}. Writing fallback text...")
            fallback = "《红楼梦》，中国古代章回体长篇小说，又名《石头记》等，被列为中国古典四大名著之首。\n" * 1000
            with open(hlm_file, 'w', encoding='utf-8') as f:
                f.write(fallback)

    with open(hlm_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Simple chunking by paragraph
    paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 5]
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p
        else:
            current_chunk += "\n" + p if current_chunk else p
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def prepare_bilingual_dataloaders(ts_samples=50000, val_ratio=0.05, max_length=256, batch_size=16):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, f"bilingual_subset_{ts_samples}.json")

    # Initialize Qwen tokenizer
    print("Loading Qwen2.5-0.5B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pad_token_id = tokenizer.pad_token_id

    if os.path.exists(cache_file):
        print(f"Loading mixed dataset from local cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
    else:
        print("Loading TinyStories dataset from Hugging Face...")
        try:
            ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
            if ts_samples and ts_samples < len(ds):
                ds = ds.select(range(ts_samples))
            ts_texts = list(ds['text'])
        except Exception as e:
            print(f"Failed to load TinyStories: {e}. Using fallback...")
            ts_texts = ["Once upon a time, there was a girl named Lily. She loved to play."] * 5000

        hlm_texts = get_hongloumeng_texts()
        
        print(f"TinyStories chunks: {len(ts_texts)}, Hongloumeng chunks: {len(hlm_texts)}")
        
        # Mix them to be roughly balanced
        mix_ratio = max(1, len(ts_texts) // max(1, len(hlm_texts)))
        hlm_texts_extended = hlm_texts * mix_ratio
        
        texts = ts_texts + hlm_texts_extended
        random.seed(42)
        random.shuffle(texts)
        
        print(f"Total mixed dataset size: {len(texts)}")
        
        print(f"Saving dataset to local cache: {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False)

    total_size = len(texts)
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size

    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    full_dataset = BilingualDataset(texts, tokenizer, max_length)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = len(tokenizer)
    # Pad to multiple of 64 for memory alignment and to safely include all special tokens
    vocab_size = ((vocab_size + 63) // 64) * 64
    return train_loader, val_loader, vocab_size, pad_token_id, tokenizer

if __name__ == "__main__":
    train_loader, val_loader, vocab_size, pad_id, tokenizer = prepare_bilingual_dataloaders(ts_samples=1000)
    print(f"Vocabulary Size: {vocab_size}, Pad ID: {pad_id}")
    for xb, yb in train_loader:
        print("Batch shape:", xb.shape)
        break
