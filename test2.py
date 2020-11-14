
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# from dataset.wmt14 import Wmt14
# data = Wmt14('cs-en')

dataset = load_dataset('wmt14', 'de-en')
trainloader = DataLoader(dataset['train'], batch_size=32, shuffle=True)
# val = next(iter(trainloader))
tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
de = map(lambda x: tokenizer(x['translation']['de'], truncation=True, padding='max_length'), trainloader)
for n in de:
    print(n)
    break

# print(list(asdf))

# for c in trainloader:
#     tokenized = tokenizer(c['translation']['cs'], truncation=True, padding='max_length')
#     input_ids = torch.tensor(tokenized['input_ids'])
#     token_type_ids = torch.tensor(tokenized['token_type_ids'])
#     attention_mask = torch.tensor(tokenized['attention_mask'])

#     break







# it = iter(trainloader)
# first = next(it)
# first_en = first['translation']['en']

