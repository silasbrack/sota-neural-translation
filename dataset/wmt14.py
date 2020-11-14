
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

def load(lang, type="torch"):
    """
    The first time you load a dataset it will take a while to download all of the files.
    It should then save it to the __pycache__ folder and then load quickly again.

    Parameters
    ----------
    lang : str
        Which languages to load in ['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en']
    type : str
        Output type selected in [None, ‘numpy’, ‘torch’, ‘tensorflow’, ‘pandas’] None means __getitem__ returns python objects (default)
    """
    dataset = load_dataset('wmt14', lang).set_format(type)
    return dataset

class Wmt14:
    def __init__(self, lang : str, type : str = 'train', batch_size : int = 32, tokenizer=None):
        self.dataset = load_dataset('wmt14', lang)
        self.loader = DataLoader(dataset[type], batch_size=32, shuffle=True)
        self.iter = iter(self.loader)
        self.tokenizer = tokenizer
        langs = lang.split('-')
        self.lang1 = langs[0]
        self.lang2 = langs[1]
