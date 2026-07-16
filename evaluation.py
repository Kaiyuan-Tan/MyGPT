import torch
from model.GPTModel import GPTModel
from config.GPT_CONFIG_124M_SHORT import GPT_CONFIG_124M
from text_generation import generate_text_simple, text_to_token_ids, token_ids_to_text
import tiktoken


inputs = torch.tensor([[16833, 3626, 6100],     # ["every effort moves",
                       [40,1107, 588]])         #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],    # [" effort moves you",
                        [588,  428,  11311]])   #  " really like chocolate"]

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(cfg=GPT_CONFIG_124M)
model.eval()

with torch.no_grad(): #A
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) # Probability of each token in vocab
token_ids = torch.argmax(probas, dim=-1, keepdim=True)

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
avg_log_probas = torch.mean(log_probas)
neg_avg_log_probas = avg_log_probas * -1

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
