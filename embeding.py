import torch
import tiktoken
from textloader import GPTDatasetV1
from torch.utils.data import DataLoader

size = 50257
dim = 256
max_length = 4

# input_ids = torch.tensor([2, 3, 5, 1])
# torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(size, dim)
# print(embedding_layer(input_ids))
# print(embedding_layer.weight)

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A 
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f: # read text file
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)