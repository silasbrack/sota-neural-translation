from torchtext.datasets import Multi30k, WMT14
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
import sys
sys.path.append("./")
import torch
import torch.nn as nn
import math
import numpy as np
# import seaborn as sns
# sns.set_palette(sns.color_palette("hls", 8))
# sns.set_context(context="poster")

def get_most_likely_words(number_of_words, output, field):
    output = torch.softmax(output, dim=0)
    values, indices = torch.topk(output, number_of_words, largest=True)
    return values, [field.vocab.itos[idx] for idx in indices]

def plot_most_likely_words(most_likely_words):
    import matplotlib.pyplot as plt
    import numpy as np
    sentence_length = len(most_likely_words)
    cols = 2
    rows = int(np.ceil(sentence_length / cols))
    fig = plt.figure()
    for i in range(sentence_length):
        probabilities, words = most_likely_words[i]
        plt.subplot(rows,cols,i+1)
        plt.bar(words, probabilities.cpu())
        plt.yticks([0.,0.2,0.4,0.6,0.8,1.])
        # axs[i].set_yticks([0.,0.2,0.4,0.6,0.8,1.])
    # fig.tight_layout()
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

    best_guesses = []
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(previous_word, hidden, encoder_outputs)
            if multiple_guesses > 1:
                best_guesses.append(get_most_likely_words(multiple_guesses, output.flatten(), english))
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    if multiple_guesses > 1:
        return translated_sentence[1:], best_guesses
    return translated_sentence[1:]

def plot_heatmap(src, trg, scores):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis')
    for (i, j), z in np.ndenumerate(scores):
        ax.text(j + 0.5, i + 0.5, '{:0.2f}'.format(z), ha='center', va='center', color='w')


    ax.set_xticklabels(trg, minor=False, rotation='horizontal')
    ax.set_yticklabels(src, minor=False)

    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    plt.show()

def plot_attention(model, sentence, german, english, device, max_length=50):
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

    attention_matrix = []
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attn = model.decoder(previous_word, hidden, encoder_outputs, return_attn=True)
            attention_matrix.append(attn.cpu().numpy()[1:])
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    attention = np.array(attention_matrix)
    plot_heatmap(translated_sentence[1:], tokens[1:], attention)

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

            # src = batch['src'].to(device)
            # trg = batch['trg'].to(device)
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

# train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))
train_data, valid_data, test_data = WMT14.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

valid_iterator, test_iterator = BucketIterator.splits(
    (valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

a = torch.load('transformer.pt', map_location=device)
model = a['model']

criterion = nn.CrossEntropyLoss()

test_loss = evaluate(model, test_iterator, criterion)
best_epoch, best_loss = a['epoch'], a['best_loss']

print(f'| Best epoch: {best_epoch}    | Best loss: {best_loss:.3f}  |')
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

test_bleu = bleu(test_data, model, SRC, TRG, device)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU : {test_bleu:.3f} |')



# sentence = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster"
# real_translation = "a man in a blue shirt is standing on a ladder and cleaning a window"
# # sentence = "ein frau mit einem orangefarbenen hut, der etwas anstarrt."
# # real_translation = "a woman in an orange hat staring at something."
# translated_sentence, best_guesses = translate_sentence(model, sentence, SRC, TRG, device, max_length=50, multiple_guesses=10)

# print(f"Translated example sentence: \n {' '.join(translated_sentence)}")
# print(f"Real example sentence: \n {real_translation}")

# plot_attention(model, sentence, SRC, TRG, device, max_length=50)

# plot_most_likely_words(best_guesses)
