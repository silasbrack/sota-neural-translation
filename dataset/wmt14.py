
from datasets import load_dataset

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

