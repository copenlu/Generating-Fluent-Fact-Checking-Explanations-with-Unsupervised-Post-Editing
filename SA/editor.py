import copy
from typing import List, Tuple

import numpy as np
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer


class RobertaEditor():
    def __init__(self):
        # TODO move these hard coded to parameters
        self.model_id = 'roberta-base'
        self.device = 'cpu'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_id)
        self.model = RobertaForMaskedLM.from_pretrained(self.model_id,
                                                        return_dict=True)
        self.ops_map = [self.insert, self.replace, self.delete]
        self.mask_vocab_idx = 50264
        print("Editor built")

    def cuda(self):
        self.model.to(self.device)

    def edit(self, inputs, ops, positions):
        edited_inputs = np.array(
            [self.ops_map[op](inp, position) for inp, op, position, in
             zip(inputs, ops, positions)])

        insert_and_replace_inputs = edited_inputs[np.where(
            ops < 2)]  # select those sentences which have mask token in them.
        insert_and_replace_outputs = self.generate(
            insert_and_replace_inputs.tolist(), np.array(positions)[np.where(
            ops < 2)])

        edited_inputs[np.where(ops < 2)] = insert_and_replace_outputs

        return edited_inputs

    def prepare_batch(self, input_tokens: List[List[str]]):
        input_texts = [' '.join(sum(instance, [])) for instance in input_tokens]
        inputs = {k: v.to(self.device) for k, v in
                  self.tokenizer(input_texts, padding=True, truncation=True,
                                 return_tensors="pt").items()}

        mask_idxs = (inputs["input_ids"] == self.mask_vocab_idx).long().max(
            dim=1).indices
        return inputs, mask_idxs

    def generate(self, input_texts, positions):
        inputs, mask_idxs = self.prepare_batch(input_texts)

        outputs = self.model(**inputs)
        generated_words = self.get_word_at_mask(outputs, mask_idxs)

        # TODO use tokenizer's mask token
        edited_text = copy.deepcopy(input_texts)
        for i in range(len(edited_text)):
            edited_text[i][positions[i][0]][positions[i][1]] = generated_words[
                i]

        return edited_text

    def insert(self, input_texts: List[List[str]], mask_idx: Tuple[int,
                                                                   int]) -> \
    List[List[str]]:
        edited_text = copy.deepcopy(input_texts)
        edited_text[mask_idx[0]] = edited_text[mask_idx[0]][:mask_idx[1]] + [
            "<mask>"] + edited_text[mask_idx[0]][mask_idx[1]:]
        return edited_text

    def replace(self, input_texts: List[List[str]],
                mask_idx: Tuple[int, int]) -> List[List[str]]:
        edited_text = copy.deepcopy(input_texts)
        edited_text[mask_idx[0]] = edited_text[mask_idx[0]][:mask_idx[1]] + [
            "<mask>"] + edited_text[mask_idx[0]][mask_idx[1] + 1:]
        return edited_text

    def delete(self, input_texts: List[List[str]], mask_idx: Tuple[int,
                                                                   int]) -> \
    List[List[str]]:
        edited_text = copy.deepcopy(input_texts)
        edited_text[mask_idx[0]].pop(mask_idx[1])
        return edited_text

    def get_word_at_mask(self, output_tensors, mask_idxs):
        mask_idxs = mask_idxs.unsqueeze(dim=1)
        return [self.tokenizer.decode(word_idx)
                for word_idx in torch.argmax(output_tensors.logits, dim=2).
                    gather(1, mask_idxs).squeeze(-1).cpu().numpy().tolist()]

    def get_contextual_word_embeddings(self, input_texts):
        inputs = {k: v.to(self.device) for k, v in
                  self.tokenizer(input_texts, padding=True,
                                 return_tensors="pt").items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 1:-1, :]

    def get_contextual_word_embeddings_sentencelevel(self, input_texts):
        inputs = {k: v.to(self.device) for k, v in
                  self.tokenizer(input_texts, padding=True,
                                 return_tensors="pt").items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :]
