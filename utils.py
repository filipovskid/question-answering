import torch
import torch.nn.functional as F


def mask_logits(logits, mask):
    mask = mask.type(torch.float32)
    return (1 - mask) * logits + mask * -1e30


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    masked_logits = mask_logits(logits, mask)
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs
