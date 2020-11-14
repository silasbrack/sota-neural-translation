from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

dataset = load_dataset('wmt14', 'de-en')
# trainloader = DataLoader(dataset['train'], batch_size=32, shuffle=True)
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# asdf = map(lambda x: tokenizer(x['translation']['de'], truncation=True, padding='max_length'), trainloader)

tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")

sentence = "Willst du einen Kaffee trinken gehen mit mir?"

input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))


