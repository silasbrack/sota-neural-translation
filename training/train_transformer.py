from torchtext.datasets import Multi30k, WMT14
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext import data
import spacy
import random
from typing import Tuple
import seaborn
import torch
from torch.autograd import Variable
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import sys
sys.path.append("./")
import models.transformer as h
from evaluation.utils import plot_training_curve

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

def train(data_iter, model, criterion, optimiser):
    model.train()
    train_loss = run_epoch((rebatch(pad_idx, b) for b in data_iter), 
                model, 
                SimpleLossCompute(model.generator, criterion, opt=optimiser))
    return train_loss
def eval(data_iter, model, criterion):
    model.eval()
    with torch.no_grad():
        eval_loss = run_epoch((rebatch(pad_idx, b) for b in data_iter), model, 
                                SimpleLossCompute(model.generator, criterion, opt=None))
    return eval_loss
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = []
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens) #/ batch.ntokens
        total_loss.append(loss.item())
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm

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
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

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

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def checkpoint_and_save(name, model):
    print('Saving.')
    torch.save(model.state_dict(), name)

def save_model_state(name, model, params, epoch, losses, bleu):
    print('Saving.')
    state = {"state_dict" : model.state_dict(), "params" : params, "epoch" : epoch, "loss": losses, "bleu" : bleu}
    torch.save(state, name)
def save_model(name, model):
    print('Saving.')
    torch.save(model, name)
def save_losses(name, d):
    torch.save(d, name)

def visualise_attention(tgt_sent, sent):
    def draw(data, x, y, ax):
        seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1,4, figsize=(16, 8))
        print("Encoder Layer", layer+1)
        for h in range(4):
            draw(model.encoder.layers[layer].self_attn.attn[0, h].data.cpu(), 
                sent, sent if h ==0 else [], ax=axs[h])
        plt.show()
        
    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1,4, figsize=(16, 8))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)].cpu(), 
                tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
        plt.show()
        print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1,4, figsize=(16, 8))
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)].cpu(), 
                sent, tgt_sent if h ==0 else [], ax=axs[h])
        plt.show()



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

MAX_LEN = 100
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),fields = (SRC, TRG)
            ,filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
# train_data, valid_data, test_data = WMT14.splits(exts = ('.de', '.en'),fields = (SRC, TRG)
#             ,filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
SRC.build_vocab(train_data.src, min_freq=2)
TRG.build_vocab(train_data.trg, min_freq=2)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
train_iter = MyIterator(train_data, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(valid_data, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)
test_iter = MyIterator(test_data, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)

args = (INPUT_DIM, OUTPUT_DIM)
kwargs = {"N" : 6}
model = h.make_model(*args, **kwargs).to(device)
print(f'The model has {count_parameters(model):,} trainable parameters.')

pad_idx = TRG.vocab.stoi["<pad>"]
criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx) # LabelSmoothing(size=OUTPUT_DIM, padding_idx=pad_idx, smoothing=0.1).to(device)
criterion_val = nn.CrossEntropyLoss(ignore_index=pad_idx)
criterion_test = nn.CrossEntropyLoss(ignore_index=pad_idx)

TO_TRAIN = False
if TO_TRAIN:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    NUM_EPOCHS = 100
    losses =  {"train": [], "val": [], "test": []}
    min_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_losses = train(train_iter, model, criterion_train, model_opt)
        losses["train"].append(train_losses)
        val_losses = eval(valid_iter, model, criterion_val)
        losses["val"].append(val_losses)

        val_loss = sum(val_losses) / len(val_losses)
        print("Validation loss for epoch", epoch+1, ":", val_loss)

        if (val_loss < min_val_loss):
            save_model_state("models/states/harvard_transformer2_state.pt", model, {"args" : args, "kwargs" : kwargs}, epoch+1, losses, None)

else:
    # model.load_state_dict(torch.load("harvard_transformer.pt"))

    # model.load_state_dict(torch.load("harvard_transformer2.pt"))
    # losses = torch.load("harvard_transformer2_loss.pt")

    state = torch.load("models/states/harvard_transformer2_state.pt", map_location=device)
    model.load_state_dict(state["state_dict"])
    losses = state["loss"]

    test_losses = eval(test_iter, model, criterion_test)
    losses["test"].append(test_losses)
    test_loss = torch.tensor(sum(test_losses) / len(test_losses))
    print(test_loss)
    print('Perplexity:', torch.exp(test_loss))

    model.eval()

    sentence = [SRC.preprocess("ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster")]
    real_translation = TRG.preprocess("a man in a blue shirt is standing on a ladder and cleaning a window")

    src = SRC.process(sentence).to(device).T
    src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TRG.vocab.stoi["<sos>"])
    translation = []
    for i in range(1, out.size(1)):
        sym = TRG.vocab.itos[out[0, i]]
        if sym == "<eos>": break
        translation.append(sym)
    print(translation)
    print(real_translation)

    plot_training_curve(losses["train"])
    plot_training_curve(losses["val"])
    
    visualise_attention(translation, real_translation)

    candidate = []
    reference = []
    for i, batch in enumerate(test_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TRG.vocab.stoi["<sos>"])

        translation = []
        for i in range(1, out.size(1)):
            sym = TRG.vocab.itos[out[0, i]]
            if sym == "<eos>": break
            translation.append(sym)
        print("Translation: \t", ' '.join(translation))
        target = []
        for i in range(1, batch.trg.size(0)):
            sym = TRG.vocab.itos[batch.trg.data[i, 0]]
            if sym == "<eos>": break
            target.append(sym)
        print("Target: \t", ' '.join(target))
        print()

        candidate.append(translation)
        reference.append([target])

    bleu = bleu_score(candidate, reference)
    print(bleu)
    # state["bleu"] = bleu
    # save_model_state("harvard_transformer2_state.pt", model, {"args" : args, "kwargs" : kwargs}, epoch+1, state["loss"], state["bleu"])

