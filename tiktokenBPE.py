import re
from importlib.metadata import version
import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")
# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of some"
# text = "HE he his her she Hi hi hello He Her, ALBBHSSDDaomcoewo, someunknownPlace, some unknow Place"
with open("the-verdict.txt", "r", encoding="utf-8") as f: # read text file
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]

context_size = 4 #A

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
# strings = tokenizer.decode(integers)
# print(strings)