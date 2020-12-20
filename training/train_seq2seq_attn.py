from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
import spacy
import torch

import random
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import sys
sys.path.append("./")
from models.seq2seq_attn import make_model
import math
import time

def checkpoint_and_save(name, model, best_loss, epoch, optimizer, epoch_loss):
    print('saving')
    print()
    state = {'model': model,'best_loss': best_loss,'epoch': epoch,'rng_state': torch.get_rng_state(), 'optimizer': optimizer.state_dict(),}
    torch.save(state, name)
    # torch.save(model.state_dict(), name)

def translate_sentence(model, sentence, german, english, device, max_length=50):
    spacy_ger = spacy.load("de_core_news_sm")

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)
    text_to_indices = [german.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(previous_word, hidden, encoder_outputs)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = []

    for _, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss.append(loss.item())

    return epoch_loss

def plot_training_curve(losses):
    loss = losses["train"]
    loss_flat = [batch for epoch in loss for batch in epoch]

    N = 16
    cumsum, moving_aves = [0], []
    for i, x in enumerate(loss_flat, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)

    plt.plot(moving_aves)
    plt.ylabel("Cross-entropy loss")
    plt.xticks(np.cumsum([len(epoch) for epoch in loss]), range(1, len(loss)+1))
    plt.xlabel("Epoch")
    plt.tight_layout
    plt.show()

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1] if prediction[-1] == "<eos>" else prediction  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = []

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss.append(loss.item())

    return epoch_loss

def save_model_state(name, model, params, epoch, losses, bleu):
    print('Saving.')
    state = {"state_dict" : model.state_dict(), "params" : params, "epoch" : epoch, "loss": losses, "bleu" : bleu}
    torch.save(state, name)

def save_model(name, model):
    print('Saving.')
    torch.save(model, name)

def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

print(INPUT_DIM, OUTPUT_DIM)

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
ATTN_DIM = 64
args = (INPUT_DIM,OUTPUT_DIM,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT,ATTN_DIM)
kwargs = {"device" : device}

model = make_model(*args, **kwargs).to(device)
model_name = "torch_Seq2Seq"

optimizer = optim.Adam(model.parameters())


print(f'The model has {count_parameters(model):,} trainable parameters')

PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


sentence1 = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster ."
sentence2 = "a man in a blue shirt is standing on a ladder and cleaning a window ."

TO_TRAIN=False
if TO_TRAIN:
    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')
    best_epoch = 0

    losses =  {"train": [], "val": [], "test": []}
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        translated_sentence1 = translate_sentence(model, sentence1, SRC, TRG, device, max_length=50)
        print(f"Translated example sentence 1: \n {' '.join(translated_sentence1)}")
        print(f"Real example sentence 1: \n {sentence2}")

        train_losses = train(model, train_iterator, optimizer, criterion, CLIP)
        train_loss = sum(train_losses) / len(train_losses)
        losses["train"].append(train_losses)
        valid_losses = evaluate(model, valid_iterator, nn.CrossEntropyLoss(ignore_index=PAD_IDX))
        valid_loss = sum(valid_losses) / len(valid_losses)
        losses["val"].append(valid_losses)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            # checkpoint_and_save('transformer.pt', model, best_valid_loss, epoch, optimizer, valid_loss) 
            save_model_state("models/states/" + model_name + ".pt", model, None, epoch+1, losses, None)
        if ((epoch - best_epoch) >= 10):
            print("no improvement in 10 epochs, break")
            break

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
else:
    state = torch.load("models/states/" + model_name + ".pt", map_location=device)
    model.load_state_dict(state["state_dict"])
    losses = state["loss"]
    train_loss = sum(losses["train"][-1])/len(losses["train"][-1])
    val_loss = sum(losses["val"][-1])/len(losses["val"][-1])

    plot_training_curve(losses)

    translated_sentence1 = translate_sentence(model, sentence1, SRC, TRG, device, max_length=50)
    print(f"Translated example sentence 1: \n {' '.join(translated_sentence1)}")
    print(f"Real example sentence 1: \n {sentence2}")

    test_losses = evaluate(model, test_iterator, criterion)
    test_loss = sum(test_losses) / len(test_losses)
    # test_bleu = bleu(test_data, model, SRC, TRG, device)

    print(f'| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} |')
    print(f'| Val   Loss: {val_loss:.3f} | Val PPL:   {math.exp(val_loss):7.3f} |')
    print(f'| Test Loss:  {test_loss:.3f} | Test PPL:  {math.exp(test_loss):7.3f} |')
    # print(f'| Test Loss:  {test_loss:.3f} | Test PPL:  {math.exp(test_loss):7.3f} | Test BLEU : {test_bleu:.3f} |')
