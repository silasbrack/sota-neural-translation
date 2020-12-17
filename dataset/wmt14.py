import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from transformers import FSMTTokenizer

class Wmt14(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        # self.train_dims = None
        self.vocab_size = 0
        self.val_fraction = 0.10
        self.max_seq_length = 32

        # self.prepare_data()
        self.setup()

    def prepare_data(self):
        # called only on 1 GPU
        load_dataset("wmt14", "de-en", "val")
        FSMTTokenizer.from_pretrained("facebook/wmt19-de-en")

    def setup(self):
        # called on every GPU
        self.dataset = load_dataset("wmt14", "de-en", "val")
        self.tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-de-en")

        self.vocab_size = self.tokenizer.vocab_size

        val_len = len(self.dataset["validation"])
        n_val = int(val_len*self.val_fraction)
        n_train = val_len - n_val

        for key in ["validation", "test"]:
            self.dataset[key] = self.dataset[key].map(self.tokenize)
        self.dataset["train"], self.dataset["validation"] = random_split(self.dataset["validation"], [n_train, n_val])
        
        # self.train_dims = next(iter(self.train_dataloader()))

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=64)

    def tokenize(self, example):
        example['translation']['de'] = self.tokenizer.encode(example['translation']['de'], return_tensors='pt', max_length=32, truncation=True, padding="max_length")[0]
        example['translation']['en'] = self.tokenizer.encode(example['translation']['en'], return_tensors='pt', max_length=32, truncation=True, padding="max_length")[0]
        return example
