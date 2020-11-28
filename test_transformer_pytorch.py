from torchtext.datasets import Multi30k, WMT14
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
import sys
sys.path.append("./")
# from models.transformer_pytorch import Encoder,Attention,Decoder,Seq2Seq
import torch
import torch.nn as nn

def get_most_likely_words(number_of_words, output, field):
    output = torch.softmax(output, dim=0)
    values, indices  = torch.topk(output, number_of_words, largest=True)
    return values, [field.vocab.itos[idx] for idx in indices]

def plot_most_likely_words(most_likely_words):
    import matplotlib.pyplot as plt
    plt.bar(most_likely_words[1], most_likely_words[0].cpu())
    plt.show()

def translate_sentence(model, sentence, german, english, device, max_length=50, multiple_guesses=0):
    import spacy
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
            if multiple_guesses > 1:
                best_guesses = get_most_likely_words(multiple_guesses, output.flatten(), english)
                plot_most_likely_words(best_guesses)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # if multiple_guesses > 1:
    #     return best_guesses
    return translated_sentence[1:]

def evaluate_bleu(model: nn.Module,iterator: BucketIterator):

    model.eval()
    corpus = []
    corpus_ref = []
    with torch.no_grad():
        
        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            print(src, trg)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

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

            corpus.append(output)
            corpus_ref.append(trg)

    score = bleu_score(corpus, corpus_ref)
    return score

def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


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

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))
# train_data, valid_data, test_data = WMT14.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

valid_iterator, test_iterator = BucketIterator.splits(
    (valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

a = torch.load('transformer.pt', map_location=device)
model = a['model']

criterion = nn.CrossEntropyLoss()
print(evaluate(model, valid_iterator, criterion))
print(evaluate(model, test_iterator, criterion))
# print(a['epoch'], a['best_loss'])
# test_bleu = bleu(test_data, model, SRC, TRG, device)
# print(test_bleu)

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

# sentence = "mein name ist sarah"
# translation = "what if it doesn't work?"

# translated_sentence = translate_sentence(model, sentence, SRC, TRG, device, max_length=50)
# print(f"Translated example sentence: \n {' '.join(translated_sentence)}")
# print(f"Real example sentence: \n {translation}")


sentence1 = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster"
sentence2 = "a man in a blue shirt is standing on a ladder and cleaning a window"
translated_sentence1 = translate_sentence(model, sentence1, SRC, TRG, device, max_length=50, multiple_guesses=10)
print(f"Translated example sentence 1: \n {' '.join(translated_sentence1)}")
print(f"Real example sentence 1: \n {sentence2}")

def plot_heatmap(src, trg, scores):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis')

    ax.set_xticklabels(trg, minor=False, rotation='vertical')
    ax.set_yticklabels(src, minor=False)

    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    plt.show()

tokens1 = ['<sos>', 'ein', 'mann', 'in', 'einem', 'blauen', 'hemd', 'steht', 'auf', 'einer', 'leiter', 'und', 'putzt', 'ein', 'fenster', '<eos>']
translated_sentence1
