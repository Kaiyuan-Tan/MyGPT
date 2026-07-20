import torch
import torch.nn as nn
import tiktoken

from config.gpt2_124m_short import GPT_CONFIG_124M
from inference.text_generation import generate_text_simple, text_to_token_ids, token_ids_to_text
from model.gpt_model import GPTModel

##################################################

tokenizer = tiktoken.get_encoding("gpt2")

##################################################

def print_gradients(model, x):
    model.zero_grad()
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])
 
    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()
 
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

torch.manual_seed(123)
model = GPTModel(cfg=GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)
# total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
# total_size_bytes = total_params * 4  #A
# total_size_mb = total_size_bytes / (1024 * 1024)  #B
# print(f"Total size of the model: {total_size_mb:.2f} MB")
