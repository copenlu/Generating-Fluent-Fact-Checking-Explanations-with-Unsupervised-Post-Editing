import copy
from typing import List, Tuple

import numpy as np
import random
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

from SA.extract_phrases import extract_phrases
from nltk.tokenize import word_tokenize


class RobertaEditor():
    def __init__(self, model_id, editor_device, max_phrases=1):

        self.model_id = model_id
        self.device = editor_device
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_id)
        self.model = RobertaForMaskedLM.from_pretrained(self.model_id,
                                                        return_dict=True).to(self.device)

        self.ops_map = [self.insert, self.reorder, self.delete]
        self.max_phrases = max_phrases
        self.mask_vocab_idx = 50264
        print("Editor built")


    def edit(self, inputs, ops):

        edited_inputs = np.array([self.ops_map[op](inp) for inp, op, in zip(inputs, ops)])
        insert_and_replace_inputs = edited_inputs[np.where(ops < 1)]  # select those sentences which have mask token in them.
        if len(insert_and_replace_inputs) > 0:#Checks if there is an op other than delete
            insert_and_replace_outputs = self.generate(insert_and_replace_inputs.tolist())
            edited_inputs[np.where(ops < 1)] = insert_and_replace_outputs

        return edited_inputs

    def prepare_batch(self, input_texts: List[str]):

        inputs = {k: v.to(self.device) for k, v in self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").items()}
        mask_idxs = (inputs["input_ids"] == self.mask_vocab_idx).long().max(dim=1).indices
        return inputs, mask_idxs

    def generate(self, input_texts):

        inputs, mask_idxs = self.prepare_batch(input_texts)
        outputs = self.model(**inputs)
        generated_words = self.get_word_at_mask(outputs, mask_idxs)

        # TODO use tokenizer's mask token
        return np.array(
            [input_text.replace(" <mask>", mask_word) for input_text, mask_word in zip(input_texts, generated_words)])


    def insert(self, input_texts: str) -> str:

        unique_phrases = extract_phrases(input_texts)
        phrases_in_input = [i for i in unique_phrases if i in input_texts]

        anchor_phrase = phrases_in_input[random.sample(range(0, len(phrases_in_input)), self.max_phrases)[0]]

        start_idx = input_texts.index(anchor_phrase)
        end_idx = start_idx + len(anchor_phrase)
        start_str = input_texts[:end_idx]
        end_str = input_texts[end_idx:]

        return start_str + " " + "<mask>" + end_str


    # def replace(self, input_texts: str, mask_idx: int) -> str:
    #     edited_text = input_texts.strip().split(" ")
    #     edited_text = edited_text[:mask_idx] + ["<mask>"] + edited_text[mask_idx+1:]
    #     return " ".join(edited_text)
    #
    #
    # def delete_word_level(self, input_texts: str, mask_idx: int) -> str:
    #     edited_text = input_texts.strip().split(" ")
    #     edited_text = edited_text[:mask_idx] + edited_text[mask_idx + 1:]
    #     return " ".join(edited_text)


    def delete(self, input_texts: str) -> str: #phrase level delete operation
        unique_phrases = extract_phrases(input_texts)
        phrases_in_input = [i for i in unique_phrases if i in input_texts]

        return [input_texts.replace(phrases_in_input[i], "") for i in random.sample(range(0, len(phrases_in_input)), self.max_phrases)][0]

    def remove_phrase(self, text, reorder_phrase):

        text = text.strip()
        text = text.replace(" " + reorder_phrase, "")
        text = text.replace(reorder_phrase, "")

        return text.strip()

    def reorder(self, input_texts: str) -> str: #phrase level delete operation

        unique_phrases = extract_phrases(input_texts)
        phrases_in_input = [i for i in unique_phrases if i in input_texts]

        reorder_phrase = phrases_in_input[random.sample(range(0, len(phrases_in_input)), self.max_phrases)[0]]

        anchor_phrase = phrases_in_input[random.sample(range(0, len(phrases_in_input)), self.max_phrases)[0]]
        while (reorder_phrase in anchor_phrase) or (anchor_phrase in reorder_phrase):
            anchor_phrase = phrases_in_input[random.sample(range(0, len(phrases_in_input)), self.max_phrases)[0]]

        start_idx = input_texts.index(anchor_phrase)
        end_idx = start_idx + len(anchor_phrase)
        start_str = self.remove_phrase(input_texts[:end_idx], reorder_phrase)
        end_str = self.remove_phrase(input_texts[end_idx:], reorder_phrase)

        return (start_str + " " + reorder_phrase + " " + end_str).strip()


    def get_word_at_mask(self, output_tensors, mask_idxs):
        mask_idxs = mask_idxs.unsqueeze(dim=1)
        return [self.tokenizer.decode(word_idx) for word_idx in torch.argmax(output_tensors.logits, dim=2).gather(1, mask_idxs).squeeze(-1).cpu().numpy().tolist()]


    def get_contextual_word_embeddings(self, input_texts):
        inputs = {k: v.to(self.device) for k, v in self.tokenizer(input_texts, padding=True, return_tensors="pt").items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 1:-1, :]


    def get_contextual_word_embeddings_sentencelevel(self, input_texts):
        inputs = {k: v.to(self.device) for k, v in self.tokenizer(input_texts, padding=True, return_tensors="pt").items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :]
