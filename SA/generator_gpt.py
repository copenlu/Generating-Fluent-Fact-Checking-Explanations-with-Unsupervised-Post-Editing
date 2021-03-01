from tqdm import tqdm
import torch
import math
import numpy as np
import os
import json
from tqdm import tqdm
import numpy as np
from nlp import load_dataset

from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration
device = 'cpu' 
model_id = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)

#to get scores sentence by sentence
def scorer(sentence):

    tokenize_input = tokenizer.tokenize(sentence, return_tensors='pt')
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    outputs=model(tensor_input, labels=tensor_input)

    loss = outputs[0]
    logits = outputs[1]

    return math.exp(loss.item()) #fluency per sentence

def scorer_batch(sentences):
    #Gpt for fluency
    tokenizer.pad_token = tokenizer.eos_token
    tensor_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    lm_labels = tensor_input["input_ids"].detach().clone()
    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

    outputs = model(input_ids=tensor_input["input_ids"],
                    attention_mask= tensor_input["attention_mask"],
                    return_dict=True)

    lm_logits = outputs.logits
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = lm_labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')  # give CE loss at each word generation step
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    prob_products_per_sample = torch.exp(-1 * loss.reshape(-1, shift_labels.shape[-1]).sum(dim=1)/(tensor_input["attention_mask"][..., 1:].contiguous().sum(dim=1)))
    return (prob_products_per_sample)
