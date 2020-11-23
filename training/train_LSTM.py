import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import pandas as pd
import spacy, random
import sys
from torchtext.data.metrics import bleu_score
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
sys.path.append("./")
from models.LSTM_baseline import DecoderLSTM, EncoderLSTM, Seq2Seq



spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_web_sm")


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
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss):
    print('saving')
    print()
    state = {'model': model,'best_loss': best_loss,'epoch': epoch,'rng_state': torch.get_rng_state(), 'optimizer': optimizer.state_dict(),}
    torch.save(state, '/content/checkpoint-NMT')
    torch.save(model.state_dict(),'/content/checkpoint-NMT-SD')

def tokenize_german(text):
  return [token.text for token in spacy_german.tokenizer(text)]

def tokenize_english(text):
  return [token.text for token in spacy_english.tokenizer(text)]

german = Field(tokenize=tokenize_german, lower=True,
               init_token="<sos>", eos_token="<eos>")

english = Field(tokenize=tokenize_english, lower=True,
               init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts = (".de", ".en"),
                                                    fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=3)
english.build_vocab(train_data, max_size=10000, min_freq=3)

print(f"Unique tokens in source (de) vocabulary: {len(german.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(english.vocab)}")




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
max_len_ger = []
max_len_eng = []
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                      batch_size = BATCH_SIZE, 
                                                                      sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.src),
                                                                      device = device)
# Dataset Sneek peek before tokenizing
count = 0
for data in train_data:
  max_len_ger.append(len(data.src))
  max_len_eng.append(len(data.trg))
  if count < 10 :
    print("German - ",*data.src, " Length - ", len(data.src))
    print("English - ",*data.trg, " Length - ", len(data.trg))
    print()
  count += 1

print("Maximum Length of English Sentence {} and German Sentence {} in the dataset".format(max(max_len_eng),max(max_len_ger)))
print("Minimum Length of English Sentence {} and German Sentence {} in the dataset".format(min(max_len_eng),min(max_len_ger)))

epoch_loss = 0.0
num_epochs = 100
best_loss = 999999
best_epoch = -1
sentence1 = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster"
ts1 = []




#CREATING THE MODEL
input_size_encoder = len(german.vocab)
target_vocab_size = len(english.vocab)
encoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
encoder_dropout = float(0.5)

encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                           hidden_size, num_layers, encoder_dropout).to(device)
input_size_decoder = len(english.vocab)
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
decoder_dropout = float(0.5)
output_size = len(english.vocab)

decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                           hidden_size, num_layers, decoder_dropout, output_size).to(device)

model = Seq2Seq(encoder_lstm,decoder_lstm,target_vocab_size,device).to(device)
print(model)

learning_rate = 0.001
writer = SummaryWriter(f"runs/loss_plot")
step = 0

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)



for epoch in range(num_epochs):
  print("Epoch - {} / {}".format(epoch+1, num_epochs))
  model.eval()
  translated_sentence1 = translate_sentence(model, sentence1, german, english, device, max_length=50)
  print(f"Translated example sentence 1: \n {translated_sentence1}")
  ts1.append(translated_sentence1)

  model.train(True)
  for batch_idx, batch in enumerate(train_iterator):
    input = batch.src.to(device)
    target = batch.trg.to(device)

    # Pass the input and target for model's forward method
    output = model(input, target)
    output = output[1:].reshape(-1, output.shape[2])
    target = target[1:].reshape(-1)

    # Clear the accumulating gradients
    optimizer.zero_grad()

    # Calculate the loss value for every epoch
    loss = criterion(output, target)

    # Calculate the gradients for weights & biases using back-propagation
    loss.backward()

    # Clip the gradient value is it exceeds > 1
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    # Update the weights values using the gradients we calculated using bp 
    optimizer.step()
    step += 1
    epoch_loss += loss.item()
    writer.add_scalar("Training loss", loss, global_step=step)

  if epoch_loss < best_loss:
    best_loss = epoch_loss
    best_epoch = epoch
    checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss) 
    if ((epoch - best_epoch) >= 10):
      print("no improvement in 10 epochs, break")
      break
  print("Epoch_Loss - {}".format(loss.item()))
  print()
  
print(epoch_loss / len(train_iterator))

score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")

