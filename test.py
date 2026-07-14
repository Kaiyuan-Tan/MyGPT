import torch
import torch.nn as nn
from model.GELU import FeedForward
from model.GELU import GELU
from model.residual import ExampleDeepNeuralNetwork
from config.GPT_CONFIG_124M import GPT_CONFIG_124M
from model.block import TransformerBlock

# ffn = FeedForward(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768)
# out = ffn(x)
# print(out.shape)

layer_sizes = [3, 3, 3, 3, 3, 1] 
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
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

x = torch.rand(2, 4, 768)  #A
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
