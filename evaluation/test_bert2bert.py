import sys
sys.path.append("./")
from datasets import load_dataset
from torchtext.datasets import Multi30k
from torchtext.data import BucketIterator
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics.metrics import bleu
import torch
import numpy as np

MAX_LEN = 100
train_data, valid_data, test_data = Multi30k()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

# dataset = load_dataset('wmt14', 'de-en')
# trainloader = DataLoader(dataset['test'], batch_size=16, shuffle=True)

# encoded = dataset_test.map(lambda example: tokenizer1([example["translation"]["en"],example["translation"]["de"]],truncation=True, padding='max_length'))


tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en").to(device)

PAD_IDX = tokenizer.pad_token_id

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
model.eval()
epoch_loss = []

with torch.no_grad():

    for _, batch in enumerate(valid_iterator):

        src = batch.src
        trg = batch.trg

        output = model.generate(src)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        epoch_loss.append(loss.item())
# candidate_corpus = []
# reference_corpus = []
# n = 0
# for val in trainloader:
#     de=val['translation']['de']
#     en=val['translation']['en']
#     # asdf = map(lambda x: tokenizer(x['translation']['de'], truncation=True, padding='max_length'), trainloader)

#     input_ids = tokenizer(de, padding=True, add_special_tokens=False, )
#     input_ids = input_ids.input_ids
#     input_ids = np.array(input_ids, dtype=np.int64)
#     np.nan_to_num(input_ids, copy=False, nan=tokenizer.unk_token_id)
#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     input_ids = input_ids.to(device)
#     output_ids = model.generate(input_ids)

#     en_out = [(tokenizer.decode(output_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)) for i in range(output_ids.shape[0])]

#     en_out_split = [s.split(' ') for s in en_out]
#     en_split = [s.split(' ') for s in en]

#     candidate = en_out_split
#     reference = [[s] for s in en_split]

#     candidate_corpus += candidate
#     reference_corpus += reference

#     n += len(de)
#     print(n)
#     print(bleu(candidate, reference))

# bs = bleu(candidate_corpus,reference_corpus)
# print(bs)

