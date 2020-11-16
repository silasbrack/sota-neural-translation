from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics.metrics import bleu

device = "cuda"

dataset = load_dataset('wmt14', 'de-en')
trainloader = DataLoader(dataset['train'], batch_size=16, shuffle=True)

# encoded = dataset_test.map(lambda example: tokenizer1([example["translation"]["en"],example["translation"]["de"]],truncation=True, padding='max_length'))


tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en").to(device)

# trainloader_iter = iter(trainloader)
for val in trainloader:

    # val = next(trainloader_iter)['translation']
    # val = next(trainloader_iter)['translation']
    de=val['translation']['de']
    en=val['translation']['en']
    # asdf = map(lambda x: tokenizer(x['translation']['de'], truncation=True, padding='max_length'), trainloader)

    input_ids = tokenizer(de, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.to(device)
    # print(input_ids)
    output_ids = model.generate(input_ids)

    en_out = [(tokenizer.decode(output_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)) for i in range(output_ids.shape[0])]

    en_out_split = [s.split(' ') for s in en_out]
    en_split = [s.split(' ') for s in en]

    candidate = en_out_split
    reference = [[s] for s in en_split]

    bs = bleu(candidate,reference)

    print(bs)

