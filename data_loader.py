"""Dataset objects and collate functions for all models and datasets."""
import json
from typing import Dict, List

import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL_IDS_LIAR = {
    'pants-fire': 2, 'barely-true': 3, 'half-true': 4, 'mostly-true': 5,
    'false': 0, 'true': 1
}
THREE_LABEL_IDS_LIAR = {
    'pants-fire': 0, 'barely-true': 1, 'mostly-true': 1, 'false': 0, 'true': 2,
    'half-true': 1
}


class LIARDataset(Dataset):
    def __init__(self, path: str, num_labels: int = 3):
        self.dataset = []
        """ The format of the instances is a dictionary with keys:
            id: str/int -- id in the corresponding dataset
            text: List[str] - main text, where the explanation apply to
            text_query: str - query or claim
            text_answer: str, optional - used for QA dataset

            label: str

            explanation_text: str, optional 
                - abstractive explanation for the instance
            explanation_sentences:  List[int], optional
                - indices of the explanation sentences from text
            explanation_sentences_hot:  List[int], optional
                - indices of the explanation sentences from text
            explanation_tokens: List[List[int]], optional
                - token indices highlighted as explanation from text
        """

        columns = ['dummy', 'id', 'statement', 'justification',
                   'ruling_without_summary', 'label', 'just_tokenized',
                   'ruling_tokenized', 'statement_tokenized', 'oracle_ids']

        df = pd.read_csv(path, sep='\t', index_col=0)
        df = df.dropna()
        df.columns = columns

        for i, row in df.iterrows():
            dict_ = dict()
            dict_['id'] = str(row['id'])

            dict_['text'] = sent_tokenize(row['ruling_without_summary'])
            dict_['text_query'] = row['statement']

            dict_['label'] = row['label']

            if num_labels == 3:
                dict_['label_id'] = THREE_LABEL_IDS_LIAR[row['label']]
            else:
                dict_['label_id'] = LABEL_IDS_LIAR[row['label']]

            dict_['text_answer'] = None

            dict_['explanation_text'] = row['justification']

            dict_['explanation_sentences'] = json.loads(row['oracle_ids'])
            dict_['explanation_sentences_hot'] = [
                1 if i in dict_['explanation_sentences']
                else 0 for i in range(len(dict_['text']))
            ]

            self.dataset.append(dict_)

    def __getitem__(self, item: int):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def get_datasets(dataset_dir: str,
                 dataset_type: str,
                 num_nabels: int = None,
                 add_logits: bool = False):
    datasets = []
    if dataset_type == 'liar':
        dataset_list = ['ruling_oracles_train.tsv',
                        'ruling_oracles_val.tsv',
                        'ruling_oracles_test.tsv']
        for ds_name in dataset_list:
            ds = LIARDataset(f'{dataset_dir}/{ds_name}',
                             num_labels=num_nabels)
            datasets.append(ds)
    elif dataset_type == 'pubmed':
        raise NotImplementedError

    if add_logits:
        with open(dataset_dir + '/logits.json') as out:
            all_logits = json.load(out)
            logits_list = [all_logits['train'],
                           all_logits['dev'],
                           all_logits['test']]

            for ds, logits in zip(datasets, logits_list):
                for i in range(len(ds)):
                    ds.dataset[i]['logits'] = logits[i]

    return datasets


def collate_explanations(instances: List[Dict],
                         tokenizer: AutoTokenizer,
                         max_length: int,
                         pad_to_max_length: bool = True,
                         device='cuda',
                         sep_sentences: bool = False,
                         cls_sentences: bool = False,
                         add_logits=False):
    """Collates a batch with data from an explanations dataset"""

    add_answer = True if 'text_answer' in instances[0] and instances[0][
        'text_answer'] != None else False
    # TODO: ensure that the answer is also here,
    #  add warnings when exceeding the length
    # [CLS] query tokens [SEP] answer [SEP] [CLS]_opt sentence1 tokens [
    # SEP]_opt [CLS]_opt sentence2 tokens ... [SEP]
    input_ids = []
    sentence_start_ids = []
    explanation_sentences = []
    query_answer_ends = []
    for instance in instances:
        instance_sentence_labels = []
        instance_sentence_starts = []
        instance_input_ids = [tokenizer.cls_token_id]

        query_tokens = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(instance['text_query']))
        instance_input_ids.extend(query_tokens)
        instance_input_ids.append(tokenizer.sep_token_id)

        if add_answer:
            tokens_answer = tokenizer.encode(instance['text_answer'],
                                             add_special_tokens=False)
            instance_input_ids += tokens_answer + [tokenizer.sep_token_id]

        query_answer_ends.append(len(instance_input_ids))

        for i, sentence in enumerate(instance['text']):
            if i in instance['explanation_sentences']:
                instance_sentence_labels.append(1)
            else:
                instance_sentence_labels.append(0)
            instance_sentence_starts.append(len(instance_input_ids))
            if cls_sentences:
                instance_input_ids.append(tokenizer.cls_token_id)

            sentence_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(sentence))
            instance_input_ids.extend(sentence_tokens)

            if sep_sentences:
                instance_input_ids.append(tokenizer.sep_token_id)

        if not sep_sentences:
            instance_input_ids.append(tokenizer.sep_token_id)

        input_ids.append(instance_input_ids)
        sentence_start_ids.append(instance_sentence_starts)
        explanation_sentences.append(instance_sentence_labels)

    if pad_to_max_length:
        batch_max_len = max_length
    else:
        batch_max_len = max([len(_s) for _s in input_ids])

    input_ids = [_s[:batch_max_len] for _s in input_ids]
    sentence_start_ids = [[i for i in ids if i < batch_max_len]
                          for ids in sentence_start_ids]
    explanation_sentences = [s_y[:len(sentence_start_ids[i])] for i, s_y in
                             enumerate(explanation_sentences)]

    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (
                batch_max_len - len(_s)) for _s in
         input_ids])

    labels = torch.tensor([_x['label_id'] for _x in instances],
                          dtype=torch.long)

    result = {
        'input_ids_tensor': padded_ids_tensor.to(device),
        'target_labels_tensor': labels.to(device),
        'sentence_start_ids': sentence_start_ids,
        'explanation_sentences_oh': explanation_sentences,
        'explanation_sentences_oh_whole': [instance['explanation_sentences_hot']
                                           for instance in instances],
        'ids': [instance['id'] for instance in instances],
        'query_answer_ends': query_answer_ends
    }

    if add_logits:
        result['logits_tensor'] = torch.tensor([instance['logits']
                                                for instance in instances]).to(
            device)

    return result


def collate_explanations_joint(instances: List[Dict],
                               tokenizer: AutoTokenizer,
                               max_length: int,
                               pad_to_max_length: bool = True,
                               device='cuda',
                               sep_sentences: bool = False,
                               cls_sentences: bool = False,
                               add_logits=False):
    result = collate_explanations(instances, tokenizer, max_length,
                                  pad_to_max_length, device, sep_sentences,
                                  cls_sentences, add_logits)

    explanation_sentences = result['explanation_sentences_oh']
    explanation_sentences_max = max([len(s) for s in explanation_sentences])
    explanation_sentences_padded = [
        s + [2] * (explanation_sentences_max - len(s)) for s in
        explanation_sentences]
    explanation_sentences_tensor = torch.tensor(explanation_sentences_padded)
    explanation_sentences_mask = explanation_sentences_tensor != 2
    explanation_sentences_tensor = explanation_sentences_tensor * \
                                   explanation_sentences_mask

    explanation_sentences_idx = result['sentence_start_ids']
    explanation_sentences_idx_max = max(
        [len(s) for s in explanation_sentences_idx])
    explanation_sentences_idx_padded = [
        s + [0] * (explanation_sentences_idx_max - len(s)) for s in
        explanation_sentences_idx]
    explanation_sentences_idx_tensor = torch.tensor(
        explanation_sentences_idx_padded, dtype=torch.long)

    result['sentences_target_tensor'] = explanation_sentences_tensor.to(device)
    result['sentences_idx_tensor'] = explanation_sentences_idx_tensor.to(device)
    result['sentences_mask'] = explanation_sentences_mask.to(device)

    query_answer_mask = []
    seq_len = result['input_ids_tensor'].size()[1]
    for query_answer_end in result['query_answer_ends']:
        query_answer_mask.append(
            [1 if t < query_answer_end else 0 for t in range(seq_len)])
    result['query_answer_mask'] = torch.tensor(query_answer_mask).to(device)

    sentence_tokens_masks = []
    for k, sentense_starts_instance in enumerate(explanation_sentences_idx):
        instance_sentences_masks = []
        for l in range(len(result['input_ids_tensor'][k]) - 1, -1, -1):
            if result['input_ids_tensor'][k][l] != tokenizer.pad_token_id:
                sentense_starts_instance.append(l + 1)
                break
        for j, sentence_start in enumerate(sentense_starts_instance[:-1]):

            sentence_mask = [0] * seq_len

            if sentence_start != 0:
                for i in range(sentence_start, sentense_starts_instance[j + 1]):
                    sentence_mask[i] = 1

            instance_sentences_masks.append(sentence_mask)

        for _ in range(
                explanation_sentences_idx_max - len(instance_sentences_masks)):
            instance_sentences_masks.append([0] * seq_len)

        sentence_tokens_masks.append(instance_sentences_masks)

    result['setences_tokens_masks'] = torch.tensor(sentence_tokens_masks,
                                                   dtype=torch.float).to(device)

    return result
