import torch
import urllib.request
from load.gpt_download import download_and_load_gpt2
from config.GPT_CONFIG_124M import model_configs, GPT_CONFIG_124M
from model.GPTModel import GPTModel
from load.load_weights import load_weights_into_gpt
import tiktoken
from text_generation import generate, text_to_token_ids, token_ids_to_text


# url = (
#     "https://raw.githubusercontent.com/rasbt/"
#     "LLMs-from-scratch/main/ch05/"
#     "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

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