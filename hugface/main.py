import torch
from xlnetmodels import XLNetModel
from tokenization_xlnet import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel()
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
print(input_ids,input_ids.shape)
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states)