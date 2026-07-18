import tiktoken
import torch
from textloader import create_dataloader_v1
from config.GPT_CONFIG_124M_SHORT import GPT_CONFIG_124M
from model.GPTModel import GPTModel
from trainer import train_model_simple, plot_losses
from pathlib import Path

tokenizer = tiktoken.get_encoding("gpt2")

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False
)

model = GPTModel(cfg=GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    device,
    num_epochs=num_epochs, 
    eval_freq=5, 
    eval_iter=1,
    start_context="Every effort moves you"
)

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

model_path = save_dir / "gpt2.pth"
torch.save(model.state_dict(), model_path)

print(f"Model saved to: {model_path}")

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)