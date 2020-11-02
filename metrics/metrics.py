import torch
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)
    
def perplexity(output, target):
    return torch.exp(cross_entropy_loss(output, target))

def bleu(output, target):
    return bleu_score(output, target)