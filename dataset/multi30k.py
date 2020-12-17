import pytorch_lightning as pl
from torchtext.datasets import Multi30k as M30k
from torchtext.data import Field, BucketIterator

class Multi30k(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        # self.train_dims = None
        self.vocab_size = 0
        self.val_fraction = 0.10
        self.max_seq_length = 32

        # self.prepare_data()
        self.setup()

    def setup(self):
        # called on every GPU
        
        self.tokenizer_de = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

        self.tokenizer_en = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

        self.train_data, self.valid_data, self.test_data = M30k.splits(exts = ('.de', '.en'),
                            fields = (self.tokenizer_de, self.tokenizer_en))
        
        self.tokenizer_de.build_vocab(self.train_data, min_freq = 2)
        self.tokenizer_en.build_vocab(self.train_data, min_freq = 2)

        self.vocab_size_de = len(self.tokenizer_de.vocab)
        self.vocab_size_en = len(self.tokenizer_en.vocab)

        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size = 64)
        
        # self.train_dims = next(iter(self.train_dataloader()))

    def train_dataloader(self):
        return self.train_iterator

    def val_dataloader(self):
        return self.valid_iterator

    def test_dataloader(self):
        return self.test_iterator



# class Multi30k(pl.LightningDataModule):

#     def __init__(self):
#         super().__init__()
#         # self.train_dims = None
#         self.vocab_size = 0
#         self.val_fraction = 0.10
#         self.max_seq_length = 32

#         # self.prepare_data()
#         self.setup()

#     def setup(self):
#         # called on every GPU
        
#         self.tokenizer_de = Field(tokenize = "spacy",
#             tokenizer_language="de",
#             init_token = '<sos>',
#             eos_token = '<eos>',
#             lower = True)

#         self.tokenizer_en = Field(tokenize = "spacy",
#             tokenizer_language="en",
#             init_token = '<sos>',
#             eos_token = '<eos>',
#             lower = True)

#         self.train_data, self.valid_data, self.test_data = M30k.splits(exts = ('.de', '.en'),
#                             fields = (self.tokenizer_de, self.tokenizer_en))
        
#         self.tokenizer_de.build_vocab(self.train_data, min_freq = 2)
#         self.tokenizer_en.build_vocab(self.train_data, min_freq = 2)

#         self.vocab_size_de = len(self.tokenizer_de.vocab)
#         self.vocab_size_en = len(self.tokenizer_en.vocab)
        
#         # self.train_dims = next(iter(self.train_dataloader()))

#     def train_dataloader(self):
#         return DataLoader(self.train_data, batch_size=64)

#     def val_dataloader(self):
#         return DataLoader(self.valid_data, batch_size=64)

#     def test_dataloader(self):
#         return DataLoader(self.test_data, batch_size=64)
