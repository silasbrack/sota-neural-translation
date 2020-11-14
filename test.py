from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer1 = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
#tokenizer2 = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")

#dataset_train = load_dataset('wmt14', "de-en")["train"]
dataset_test = load_dataset('wmt14', "de-en")["test"]

tokenizer1.model_max_length = 64
encoded = dataset_test.map(lambda example: tokenizer1([example["translation"]["en"],example["translation"]["de"]],truncation=True, padding='max_length'))
dataloader_test = torch.utils.data.DataLoader(encoded, batch_size=32)
