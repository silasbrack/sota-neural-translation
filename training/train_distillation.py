import sys
sys.path.append("./")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel, BertTokenizer
from torchtext.datasets import Multi30k
import torch
import torch.nn.functional as F

def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

german = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
english = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en").to(device)

sentence = "Willst du einen Kaffee trinken gehen mit mir?"
inputs = german(sentence, return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
logits, a = model(input_ids=inputs.input_ids, decoder_input_ids=labels)
print(logits.shape, a.shape)
out = model.generate(input_ids=inputs.input_ids, output_hidden_states=True)
print(out)
# print(english.decode(model.generate(), skip_special_tokens=True))
# model.

# print(model(input_ids=input_ids))
# print(tokenizer.decode(output_ids, skip_special_tokens=True))

# # student_model =
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# logits, _, _ = model(input_ids=input_ids, decoder_input_ids=input_ids, output_attentions=True)

# print(F.softmax(logits, dim=2))
# output_ids = torch.argmax(logits, dim=2)
# print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# # generation
# generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
# print(tokenizer.decode(generated[0], skip_special_tokens=True))
