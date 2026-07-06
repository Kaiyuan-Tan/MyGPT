import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],# Your    (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one     6 (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)
# attention for a single word
query = inputs[1] # journey
attn_scores_2 = torch.empty(inputs.shape[0])
for i ,x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print(attn_weights_2)

context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2 += x_i*attn_weights_2[i]

# attention for all inputs
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=1)
    # inputs.shape = [6, 3]
    # attn_weights.shape = [6, 6]
all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)

x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query 
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

keys = inputs @ W_key 
values = inputs @ W_value

keys_2 = keys[1] #A
attn_score_22 = query_2.dot(keys_2)

attn_scores_2 = query_2 @ keys.T # All attention scores for given query

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
context_vec_2 = attn_weights_2 @ values

print(context_vec_2)
