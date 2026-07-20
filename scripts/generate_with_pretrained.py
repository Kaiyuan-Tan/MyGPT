import torch
import tiktoken
from pathlib import Path

from config.gpt2_124m import GPT_CONFIG_124M, model_configs
from inference.text_generation import generate, text_to_token_ids, token_ids_to_text
from model.gpt_model import GPTModel
from pretrained.gpt2_download import download_and_load_gpt2
from pretrained.weight_loader import load_weights_into_gpt


# url = (
#     "https://raw.githubusercontent.com/rasbt/"
#     "LLMs-from-scratch/main/ch05/"
#     "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir=PROJECT_ROOT / "gpt2",
)

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
gpt = GPTModel(cfg=NEW_CONFIG)

load_weights_into_gpt(gpt, params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)
tokenizer = tiktoken.get_encoding("gpt2")

# torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Thank you GPT", tokenizer),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
    )

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
