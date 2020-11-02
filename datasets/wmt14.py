
from datasets import load_dataset
# from torch.utils.data import Dataset

def load(lang='fr-en'):
    dataset = load_dataset('wmt14', lang)

# train, val, test = WMT14.splits(exts=('.de', '.en'), fields=(DE, EN))
