# Project2 Plan — TinyStories GPT Pretraining

目标：
- 在 karpathy/tinystories-gpt4-clean 上进行自回归语言模型训练（小型 GPT）。
- 提供可复现的训练脚本、配置与评估流程。

高层步骤（优先级顺序）：
1. 数据准备
   - 使用 dataset2.py 加载并本地缓存数据（已实现）。
   - 验证 DataLoader 输出形状，测试单 batch 前向/反向。

2. 模型
   - 选项 A（推荐快速）：使用 transformers.GPT2Config + GPT2LMHeadModel，small config（n_layer=6,n_head=8, n_embd=512）。
   - 选项 B：实现轻量级 Transformer LM（仅在需要自定义时）。

3. 配置与超参（config.yaml）
   - max_length: 256
   - batch_size: 16（或按显存调整）
   - lr: 2e-4
   - weight_decay: 0.01
   - epochs / max_steps: 指定之一
   - seed, save_interval_steps, eval_interval_steps

4. 训练脚本（train.py）
   - 构建模型、optimizer（AdamW）、scheduler（线性预热+衰减）
   - 支持混合精度（torch.cuda.amp）
   - 训练/验证循环，记录 loss、perplexity
   - 定期保存检查点（包括 optimizer 状态和 step）

5. 评估与生成（eval.py / utils）
   - 验证集困惑度计算
   - 简单文字生成：top-k/top-p 解码函数
   - 保存示例到文件以便人工评估

6. 日志与可视化
   - 最小：每 eval_interval 写入 JSON/CSV
   - 可选：tensorboard or wandb

7. 实验记录与复现
   - 使用 config 文件记录所有超参
   - 固定随机种子（torch, numpy, random）
   - 保存模型与生成样例

8. 快速 smoke 测试
   - 运行小样本（max_samples=1000）、少步数（100 steps）确保无错误并能生成文本

运行示例（本地 GPU）：
1. 准备数据（已缓存）
   python -c "from dataset2 import prepare_dataloaders; prepare_dataloaders(max_samples=5000, batch_size=8)"
2. 训练（伪命令）：
   python train.py --config config.yaml
3. 验证/生成：
   python eval.py --ckpt checkpoints/step-1000.pt --prompt "Once upon a time"

风险与注意事项：
- 显存：根据模型大小调整 batch size 或使用梯度累积。
- Tokenizer pad_token 已设置为 eos_token，训练时注意忽略 pad 在 loss 计算上的影响（可用 attention_mask）。
- 保存与加载时确保 tokenizer 也被保存以复现生成。

交付物：
- train.py, eval.py, utils.py, config.yaml, PROJECT2_PLAN.md, README.md, checkpoints/
- 一个短训练的 demo checkpoint 与生成示例

短期目标（第一天内）：
- 确认 dataset2.py 能产出正确 tensor，完成 smoke 测试并运行 100 步训练。

