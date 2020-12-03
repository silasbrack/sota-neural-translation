import sys
sys.path.append("./")
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
from utils import plot_training_curve,plot_loss_curves
from torch import nn
import torch
import time
import matplotlib.pyplot as plt
import seaborn

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(vars(new)["src"]))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(vars(new)["trg"]) + 2)
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
def visualise_attention(tgt_sent, sent):
    def draw(data, x, y, ax):
        seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1,4, figsize=(16, 6))
        print("Encoder Layer", layer+1)
        for h in range(4):
            draw(model.encoder.layers[layer].self_attn.attn[0, h].data.cpu(), 
                sent, sent if h ==0 else [], ax=axs[h])
        plt.show()
        
    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1,4, figsize=(16, 6))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)].cpu(), 
                tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
        plt.show()
        print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1,4, figsize=(16, 6))
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)].cpu(), 
                sent, tgt_sent if h ==0 else [], ax=axs[h])
        plt.show()
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
def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
def evaluate(data_iter, model, criterion):
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

model_name = "harvard_transformer2_state"
args = (INPUT_DIM, OUTPUT_DIM)
kwargs = {"N" : 6}
model = h.make_model(*args, **kwargs).to(device)

state = torch.load(model_name + ".pt")
model.load_state_dict(state["state_dict"])
losses = state["loss"]

pad_idx = TRG.vocab.stoi["<pad>"]
criterion_test = nn.CrossEntropyLoss(ignore_index=pad_idx)

test_losses = evaluate(test_iter, model, criterion_test)
losses["test"].append(test_losses)
test_loss = torch.tensor(sum(test_losses) / len(test_losses))
print(test_loss)
print('Perplexity:', torch.exp(test_loss))

sentence = [SRC.preprocess("ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster")]
real_translation = TRG.preprocess("a man in a blue shirt is standing on a ladder and cleaning a window")

src = SRC.process(sentence).to(device).T
src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
model.eval()
out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TRG.vocab.stoi["<sos>"])
translation = []
for i in range(1, out.size(1)):
    sym = TRG.vocab.itos[out[0, i]]
    if sym == "<eos>": break
    translation.append(sym)
print(translation)
print(real_translation)

# plot_loss_curves(losses["train"], losses["val"])

# visualise_attention(translation, real_translation)

# candidate = []
# reference = []
# for i, batch in enumerate(test_iter):
#     src = batch.src.transpose(0, 1)[:1]
#     src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
#     model.eval()
#     out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TRG.vocab.stoi["<sos>"])

#     translation = []
#     for i in range(1, out.size(1)):
#         sym = TRG.vocab.itos[out[0, i]]
#         if sym == "<eos>": break
#         translation.append(sym)
#     print("Translation: \t", ' '.join(translation))
#     target = []
#     for i in range(1, batch.trg.size(0)):
#         sym = TRG.vocab.itos[batch.trg.data[i, 0]]
#         if sym == "<eos>": break
#         target.append(sym)
#     print("Target: \t", ' '.join(target))
#     print()

#     candidate.append(translation)
#     reference.append([target])

# score = bleu(candidate, reference)
# print(score)
# # state["bleu"] = bleu
# # save_model_state("harvard_transformer2_state.pt", model, {"args" : args, "kwargs" : kwargs}, epoch+1, state["loss"], state["bleu"])


# dataset = load_dataset('wmt14', 'de-en', 'test')['test']['translation']
# trainloader = DataLoader(dataset, batch_size=16, shuffle=True)

# model.eval()

# candidate = []
# reference = []
# for val in trainloader:
#     de=val['de']
#     en=val['en']

#     de_tokens = [SRC.preprocess(sentence) for sentence in de]
#     en_tokens = [TRG.preprocess(sentence) for sentence in en]
#     src = SRC.process(de_tokens).to(device).T[:1]
#     trg = TRG.process(en_tokens).to(device).T[:1]
#     src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
#     out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TRG.vocab.stoi["<sos>"])

#     translation = []
#     for i in range(1, out.size(1)):
#         sym = TRG.vocab.itos[out[0, i]]
#         if sym == "<eos>": break
#         translation.append(sym)
#     target = []
#     for i in range(1, trg.size(1)):
#         sym = TRG.vocab.itos[trg[0, i]]
#         if sym == "<eos>": break
#         target.append(sym)
#     candidate.append(translation)
#     reference.append([target])

# print(bleu(candidate, reference))
