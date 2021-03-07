from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class NLIScorer():
    def __init__(self, device):
        hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3" \
                            "-nli"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hg_model_hub_name).to(self.device)
        self.max_length = 256

    def __call__(self, original_text, new_text):
        tokenized_input_seq_pair = self.tokenizer.encode_plus(original_text, new_text,
                                                         max_length=self.max_length,
                                                         return_token_type_ids=True,
                                                         truncation=True)

        input_ids = torch.Tensor(
            tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        token_type_ids = torch.Tensor(
            tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(
            tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

        outputs = self.model(input_ids.to(self.device),
                        attention_mask=attention_mask.to(self.device),
                        token_type_ids=token_type_ids.to(self.device),
                        labels=None)

        return 1 - torch.softmax(outputs[0], dim=1)[0].detach().cpu().tolist()[2]


def compute_nli_score(premise, hypothesis):

    max_length = 256

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    entailment_score = predicted_probability[0]
    neutral_score = predicted_probability[1]
    contradiction_score = predicted_probability[2]

    print("Entailment:", entailment_score)
    print("Neutral:", neutral_score)
    print("Contradiction:", contradiction_score)

    return entailment_score, neutral_score, contradiction_score


def nli_scorer(new_justs, org_justs):
    print("all_new_justs", new_justs)
    print("all_org_justs", org_justs)
    entail_scores = []
    neutral_scores = []
    contradict_scores = []

    for new, org in zip(new_justs, org_justs):
        entail_just = []
        neutral_just = []
        contradict_just = []
        new = new.split("?")
        for sent1, sent2 in zip(new.split("."), org.split(".")):

            entailment_score_sent, neutral_score_sent, contradiction_score_sent = compute_nli_score(sent1, sent2)

            entail_just.append(entailment_score_sent)
            neutral_just.append(neutral_score_sent)
            contradict_just.append(contradiction_score_sent)

        entail_scores.append(np.mean(entail_just))
        neutral_scores.append(np.mean(neutral_just))
        contradict_scores.append(np.mean(contradict_just))

    return entail_scores, neutral_scores, contradict_scores


if __name__== "__main__":

    premise = "Two women are embracing while holding to go packages."
    hypothesis = "Two women are embracing while holding to go packages."
    sc = compute_nli_score(premise, hypothesis)
    print(sc)