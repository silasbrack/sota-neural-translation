from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics.metrics import bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset = load_dataset('wmt14', 'de-en')
# trainloader = DataLoader(dataset['train'], batch_size=16, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en").to(device)