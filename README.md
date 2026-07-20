# MyGPT

MyGPT is an educational PyTorch implementation of a GPT-2-style language model. It includes tokenization, data loading, causal self-attention, Transformer blocks, training, text generation, and GPT-2 weight loading.

## Project Structure

```text
MyGPT/
├── config/                      # GPT-2 model configurations
│   ├── gpt2_124m.py             # Standard 1,024-token context
│   └── gpt2_124m_short.py       # Short 256-token training context
├── data/
│   └── the_verdict.txt          # Sample training corpus
├── experiments/                 # Learning examples and smoke checks
├── inference/
│   └── text_generation.py       # Greedy, temperature, and top-k generation
├── model/                       # Core model components
│   ├── feed_forward.py
│   ├── gpt_model.py
│   ├── layer_norm.py
│   ├── multi_head_attention.py
│   ├── self_attention.py
│   └── transformer_block.py
├── pretrained/                  # GPT-2 download and weight conversion
├── scripts/                     # Runnable entry points
│   ├── generate_with_pretrained.py
│   └── train_from_scratch.py
├── training/                    # Dataset, loss, and training utilities
├── checkpoints/                 # Local checkpoints, ignored by Git
└── gpt2/                        # Downloaded GPT-2 files, ignored by Git
```

Run modules from the project root with `python -m`.

## Environment

- Python 3.10+
- PyTorch
- `tiktoken`
- Optional CUDA-compatible GPU

Install the base dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch tiktoken matplotlib
```

For GPT-2 weight download and conversion:

```powershell
python -m pip install numpy requests tqdm tensorflow
```

## Train from Scratch

```powershell
python -m scripts.train_from_scratch
```

The script:

1. Tokenizes `data/the_verdict.txt` with GPT-2 BPE.
2. Uses a 90%/10% train-validation split.
3. Trains the model with AdamW.
4. Prints losses and generated samples.
5. Saves weights to `checkpoints/gpt2.pth`.

Edit `scripts/train_from_scratch.py` and `config/gpt2_124m_short.py` to change training settings.

## Generate with Pretrained GPT-2 Weights

```powershell
python -m scripts.generate_with_pretrained
```

The first run downloads GPT-2 124M TensorFlow weights to `gpt2/124M/`, converts them, and generates text from `Thank you GPT`.


## Experiments

| Command | Purpose |
| --- | --- |
| `python -m experiments.simple_tokenizer` | Build and use a basic tokenizer |
| `python -m experiments.tiktoken_bpe` | Inspect GPT-2 BPE context-target pairs |
| `python -m experiments.token_embeddings` | Check token and position embedding shapes |
| `python -m experiments.attention_calculation` | Calculate scaled dot-product attention |
| `python -m experiments.top_k_sampling` | Inspect top-k probability filtering |
| `python -m experiments.cross_entropy_loss` | Calculate language-model cross-entropy loss |
| `python -m experiments.model_smoke_test` | Build the full model and generate text |

`dummy_gpt_model.py` and `residual_connections.py` contain supporting definitions for learning experiments.

## Model Configurations

| Setting | From-scratch training | Pretrained GPT-2 |
| --- | ---: | ---: |
| Vocabulary size | 50,257 | 50,257 |
| Context length | 256 | 1,024 |
| Embedding size | 768 | 768 |
| Attention heads | 12 | 12 |
| Transformer layers | 12 | 12 |
| Dropout | 0.1 | 0.1 |
| QKV bias | No | Yes |

## Generation APIs

- `generate_text_simple`: greedy decoding.
- `generate`: temperature sampling with optional top-k filtering.

## Notes

- The sample corpus is too small for general-purpose text generation.
- Some experiments instantiate the full 124M model and require substantial memory.
- Training settings are defined directly in configuration and entry-point files.
- Older PyTorch versions may not support `torch.load(..., weights_only=True)`.

## Acknowledgments

The learning path and GPT-2 weight-loading code are based on Sebastian Raschka's *Build a Large Language Model (From Scratch)* and [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).
