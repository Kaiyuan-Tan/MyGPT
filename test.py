import torch
import torch.nn as nn
from model.GELU import FeedForward
from model.GELU import GELU
from model.residual import ExampleDeepNeuralNetwork
from config.GPT_CONFIG_124M import GPT_CONFIG_124M
from model.block import TransformerBlock
import tiktoken
from model.GPTModel import GPTModel
from text_generation import generate_text_simple

# ffn = FeedForward(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768)
# out = ffn(x)
# print(out.shape)

# layer_sizes = [3, 3, 3, 3, 3, 1] 
# sample_input = torch.tensor([[1., 0., -1.]])

##################################################

tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

##################################################

# torch.manual_seed(123)
# model1 = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
# model2 = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

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

# print_gradients(model2, sample_input)

# x = torch.rand(2, 4, 768)  #A
# block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)
# print("Input shape:", x.shape)
# print("Output shape:", output.shape)


model = GPTModel(cfg=GPT_CONFIG_124M)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)
# total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
# total_size_bytes = total_params * 4  #A
# total_size_mb = total_size_bytes / (1024 * 1024)  #B
# print(f"Total size of the model: {total_size_mb:.2f} MB")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A

model.eval()
out = generate_text_simple(
    model = model,
    idx = encoded_tensor,
    max_new_tokens = 6,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)