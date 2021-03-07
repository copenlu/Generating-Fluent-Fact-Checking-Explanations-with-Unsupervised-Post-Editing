import torch
import math

from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration


class GPT2FluencyScorer():
    def __init__(self, model_id, fluency_device):

        self.model_id = model_id
        self.device = fluency_device
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)

        #to get scores sentence by sentence
    def scorer(self, sentence):

        tokenize_input = self.tokenizer.tokenize(sentence, return_tensors='pt')
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)]).to(self.device)
        outputs = self.model(tensor_input, labels=tensor_input)

        loss = outputs[0]
        logits = outputs[1]

        return math.exp(loss.item()) #fluency per sentence

    def scorer_batch(self, sentences):
        #Gpt for fluency
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tensor_input = {k: v.to(self.device) for k,v in self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').items()}


        lm_labels = tensor_input["input_ids"].detach().clone()
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.model(input_ids=tensor_input["input_ids"],
                    attention_mask= tensor_input["attention_mask"],
                    return_dict=True)

        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')  # give CE loss at each word generation step
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        prob_products_per_sample = torch.exp(-1 * loss.reshape(-1, shift_labels.shape[-1]).sum(dim=1)/(tensor_input["attention_mask"][..., 1:].contiguous().sum(dim=1))).cpu()
        return (prob_products_per_sample)
