# MyGPT

MyGPT 是一个使用 PyTorch 从零实现 GPT-2 风格语言模型的学习项目。项目覆盖了文本分词、训练样本构造、因果自注意力、Transformer、损失计算、模型训练、文本生成，以及加载 OpenAI GPT-2 预训练权重的完整流程。

本项目适合用来理解 GPT 的内部结构和训练过程，定位是教学与实验，而不是生产级训练框架。

## 功能

- 使用 GPT-2 BPE 分词器处理文本；
- 从零实现 token embedding 与位置编码；
- 实现带因果掩码的多头自注意力；
- 实现 LayerNorm、GELU、前馈网络与残差连接；
- 组合完整的 GPT-2 风格 Transformer 模型；
- 支持训练集/验证集损失评估；
- 支持贪心解码、temperature 和 top-k 采样；
- 支持下载并加载 OpenAI GPT-2 预训练权重。

## 模型配置

项目主要使用 GPT-2 Small（124M 规格）的结构配置：

| 配置项 | 从头训练 | GPT-2 预训练权重 |
| --- | ---: | ---: |
| 词表大小 | 50,257 | 50,257 |
| 上下文长度 | 256 | 1,024 |
| 嵌入维度 | 768 | 768 |
| 注意力头数 | 12 | 12 |
| Transformer 层数 | 12 | 12 |
| Dropout | 0.1 | 0.1 |
| QKV bias | 否 | 是 |

两套配置分别位于 `config/GPT_CONFIG_124M_SHORT.py` 和 `config/GPT_CONFIG_124M.py`。

## 项目结构

```text
MyGPT/
├── config/                 # GPT 模型配置
├── load/                   # GPT-2 权重下载、转换与加载
├── model/                  # 注意力、Transformer Block、GPTModel 等
├── checkpoints/            # 从头训练生成的模型检查点
├── the-verdict.txt         # 示例训练语料
├── textloader.py           # 滑动窗口数据集与 DataLoader
├── loss.py                 # 交叉熵损失计算
├── trainer.py              # 训练、验证和样例生成循环
├── text_generation.py      # 贪心、temperature、top-k 文本生成
├── train.py                # 从头训练入口
├── pretrain.py             # 加载 GPT-2 预训练权重并生成文本
├── tokenizer.py            # 简易分词器实验
└── test.py                 # 模型组件与生成实验
```

`attention_test.py`、`embeding.py`、`randomize.py`、`evaluation.py` 等文件是对注意力、嵌入、采样和损失计算的分步实验。

## 环境要求

- Python 3.10 或更高版本；
- PyTorch；
- 可选的 CUDA GPU。没有 GPU 时会自动使用 CPU，但完整模型训练会很慢。

建议在虚拟环境中安装依赖。PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch tiktoken matplotlib
```

如果需要运行 GPT-2 预训练权重下载与转换，再安装：

```powershell
pip install numpy requests tqdm tensorflow
```

> PyTorch 的安装命令会因操作系统和 CUDA 版本而异；需要 GPU 支持时，请使用与本机 CUDA 环境匹配的 PyTorch 版本。

## 快速开始

所有命令都应在项目根目录执行。

### 1. 从头训练

```powershell
python train.py
```

该脚本会：

1. 使用 `tiktoken` 对 `the-verdict.txt` 编码；
2. 按 90%/10% 划分训练集和验证集；
3. 使用 AdamW 训练 10 个 epoch；
4. 定期输出训练损失、验证损失和生成样例；
5. 将模型保存到 `checkpoints/gpt2.pth`。

训练超参数可以直接在 `train.py` 中调整，包括 batch size、学习率、epoch 数、上下文长度和评估频率。

### 2. 使用已训练检查点生成文本

```python
import torch
import tiktoken

from config.GPT_CONFIG_124M_SHORT import GPT_CONFIG_124M
from model.GPTModel import GPTModel
from text_generation import generate, text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

model = GPTModel(GPT_CONFIG_124M)
state_dict = torch.load(
    "checkpoints/gpt2.pth",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=50,
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=1.0,
    top_k=50,
)

print(token_ids_to_text(token_ids, tokenizer))
```

### 3. 加载 OpenAI GPT-2 权重

```powershell
python pretrain.py
```

首次运行时，脚本会把 GPT-2 124M 的 TensorFlow 格式权重下载到 `gpt2/124M/`，转换并加载到本项目实现的 `GPTModel` 中，随后从 `Thank you GPT` 开始生成文本。

下载文件和模型检查点体积较大，请预留足够的磁盘空间与内存。模型下载需要网络连接。

## 文本生成策略

`text_generation.py` 提供两种生成接口：

- `generate_text_simple`：每一步选择概率最高的 token，输出稳定但较单一；
- `generate`：支持 temperature 与 top-k，可控制随机性和候选词范围。

通常，较低的 temperature 会让输出更确定，较高的 temperature 会增加多样性；减小 top-k 会限制每一步参与采样的候选 token 数。

## 注意事项

- 示例语料很小，从头训练的结果主要用于验证流程，无法达到通用 GPT-2 的生成质量；
- 脚本使用相对路径，请从项目根目录运行；
- 当前项目没有统一的命令行参数或依赖锁定文件，实验参数主要在各脚本中修改；
- `checkpoints/` 和常见模型检查点扩展名已在 `.gitignore` 中忽略；
- 若使用旧版 PyTorch，`torch.load` 可能不支持 `weights_only` 参数，此时可删除该参数。

## 致谢

项目的学习路径和部分 GPT-2 权重下载代码参考了 Sebastian Raschka 的 *Build a Large Language Model (From Scratch)* 及其开源项目 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
