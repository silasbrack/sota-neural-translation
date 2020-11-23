import datasets
import spacy
from torchtext import data, datasets
import pickle
import torch

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_de, init_token = BOS_WORD, 
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)
                 
MIN_FREQ = 15

def tokenize_to_file(dataset, source : data.Field, target : data.Field, file_path : str = ""):
    import dill

    MAX_LEN = 20
    train, val, test = dataset.splits(
        exts=('.en', '.de'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN 
        and len(vars(x)['trg']) <= MAX_LEN)

    name = train.name
    
    train_en = [x.src for x in train]
    train_de = [x.trg for x in train]

    val_en = [x.src for x in val]
    val_de = [x.trg for x in val]

    test_en = [x.src for x in test]
    test_de = [x.trg for x in test]

    dict_en, vocab_en = build_vocab(train['en'], SRC, MIN_FREQ)
    dict_de, vocab_de = build_vocab(train['de'], TGT, MIN_FREQ)
    
    train_en = [torch.tensor([dict_en(word) for word in sentence]) for sentence in train_en] 
    train_de = [torch.tensor([dict_de(word) for word in sentence]) for sentence in train_de] 
    
    val_en = [torch.tensor([dict_en(word) for word in sentence]) for sentence in val_en] 
    val_de = [torch.tensor([dict_de(word) for word in sentence]) for sentence in val_de] 
    
    test_en = [torch.tensor([dict_en(word) for word in sentence]) for sentence in test_en] 
    test_de = [torch.tensor([dict_de(word) for word in sentence]) for sentence in test_de] 
    
    train = { 'en' : train_en, 'de' : train_de }
    val = { 'en' : val_en, 'de' : val_de }
    test = { 'en' : test_en, 'de' : test_de }   
    
    with open(file_path + name + '.train', 'wb') as handle:
        dill.dump(train, handle)
    with open(file_path + name + '.val', 'wb') as handle:
        dill.dump(val, handle)
    with open(file_path + name + '.test', 'wb') as handle:
        dill.dump(test, handle)

def load_from_file(split : str, file_path : str = ""):
    import dill
    with open(file_path + split, 'rb') as handle:
        return dill.load(handle)

def load(split : str, file_path = ""):
    from os import path
    if not path.exists(file_path + split):    
        tokenize_to_file(datasets.IWSLT, SRC, TGT, file_path=".data/")
    return load_from_file(split, file_path)

def build_vocab(data, field, min_freq : int = 0):
    from collections import defaultdict
    field.build_vocab(data, min_freq=min_freq)
    vocab = field.vocab.itos
    dictionary = defaultdict(lambda: "<UNK>")
    for i, word in enumerate(vocab):
        dictionary[word] = vocab
    return dictionary, vocab


train = load('iwslt.train', file_path=".data/")

# for i in range(5):
#     print(f"Example={i}:{(train['en'][i],train['de'][i])}")

a = train['en'][25]
print(torch.tensor([dictionary[word] for word in a]))



