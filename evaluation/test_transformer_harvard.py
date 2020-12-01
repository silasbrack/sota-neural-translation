from torchtext.datasets import Multi30k
from torchtext.data import Field
from torchtext import data
import pickle
import models.harvard_transformer as h
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from metrics.metrics import bleu
import numpy as np
from torch.autograd import Variable

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new["de"]))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new["en"]) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

SRC = Field(tokenize = "spacy",
            tokenizer_language="de_core_news_sm",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en_core_web_sm",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MAX_LEN = 100
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),fields = (SRC, TRG)
            ,filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
SRC.build_vocab(train_data.src, min_freq=2)
TRG.build_vocab(train_data.trg, min_freq=2)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

model = h.make_model(INPUT_DIM, OUTPUT_DIM, N=6).to(device)
model.load_state_dict(torch.load("sota-neural-translation\\harvard_transformer.pt"))

dataset = load_dataset('wmt14', 'de-en', 'test')['test']['translation']
trainloader = DataLoader(dataset, batch_size=16, shuffle=True)

model.eval()

candidate = []
reference = []
for val in trainloader:
    de=val['de']
    en=val['en']

    de_tokens = [SRC.preprocess(sentence) for sentence in de]
    en_tokens = [TRG.preprocess(sentence) for sentence in en]
    src = SRC.process(de_tokens).to(device).T[:1]
    trg = TRG.process(en_tokens).to(device).T[:1]
    src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TRG.vocab.stoi["<sos>"])

    translation = []
    for i in range(1, out.size(1)):
        sym = TRG.vocab.itos[out[0, i]]
        if sym == "<eos>": break
        translation.append(sym)
    target = []
    for i in range(1, trg.size(1)):
        sym = TRG.vocab.itos[trg[0, i]]
        if sym == "<eos>": break
        target.append(sym)
    candidate.append(translation)
    reference.append([target])

print(bleu(candidate, reference))
