import argparse
import json
import random
import os
from collections import defaultdict
from functools import partial
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import torch
from captum.attr import IntegratedGradients, InputXGradient
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from data_loader import collate_explanations_joint, get_datasets
from saliency.pretrain_longformer import BertLongForMaskedLM
from saliency.model_builder import ExtractClassifyJointModel, ClassifyExtractJointModel, ClassifierModel


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, batch):
        output = self.model(tokens, batch)
        return output[0]


def get_model_embedding_emb(model):
    if args.model == 'trans':
        return model.model.transformer_model.embeddings.embedding.word_embeddings
    else:
        return model.embedding.embedding


def summarize_attributions(attributions, type='mean', model=None, tokens=None):
    if type == 'none':
        return attributions
    elif type == 'dot':
        embeddings = get_model_embedding_emb(model)(tokens)
        attributions = torch.einsum('bwd, bwd->bw', attributions, embeddings)
    elif type == 'mean':
        attributions = attributions.mean(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
    elif type == 'l2':
        attributions = attributions.norm(p=2, dim=-1)
        if len(attributions.size()) > 2:
            attributions = attributions.squeeze(0)

    return attributions


def eval_sentence_saliency(model,
                           dl,
                           ds,
                           token_aggregation,
                           cls_aggregation,
                           sentence_aggregation,
                           labels,
                           n_sentences=1):
    model.train()

    ablator = InputXGradient(model)
    token_ids = []
    class_attr_list = defaultdict(lambda: [])
    pred_classes = []

    for batch in tqdm(dl):
        token_ids += batch['input_ids_tensor'].detach().cpu().numpy().tolist()

        input_embeddings = model.transformer_model.embeddings(batch['input_ids_tensor'])
        for cls_ in range(labels):
            attributions = ablator.attribute(input_embeddings, target=cls_,
                                             additional_forward_args=(batch,))

            attributions = summarize_attributions(attributions,
                                                  type=token_aggregation,
                                                  model=model,
                                                  tokens=batch['input_ids_tensor']).detach().cpu(

            ).numpy().tolist()
            class_attr_list[cls_] += attributions

        if cls_aggregation == 'pred_cls':
            pred_classes += torch.argmax(model(batch['input_ids_tensor']), dim=-1).detach().cpu().numpy().tolist()

    # [CLS] query tokens [SEP] answer [SEP]
    # [CLS]_opt sentence1 tokens [SEP]_opt
    # [CLS]_opt sentence2 tokens ...
    # [SEP]
    sent_scores, word_scores = [], []
    model_specific_path = f'data/{args.model_path}_{args.token_aggregation}_{args.cls_aggregation}_{args.sentence_aggregation}'
    output_file_sentences = f'{model_specific_path}/sentence_scores_{args.mode}.jsonl'
    output_file_words = f'{model_specific_path}/word_scores_{args.mode}.jsonl'
    os.makedirs(model_specific_path, exist_ok=True)

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    with open(output_file_sentences, 'w') as out_s, open(output_file_words, 'w') as out_w:

        for instance_i, instance in enumerate(ds):
            sent_json = {'id': instance['id'], 'sentence_scores': []}
            word_json = {'id': instance['id'], 'word_scores': []}

            saliencies = []
            current_sent_sal = []
            token_ids[instance_i] = [tok for tok in token_ids[instance_i] if
                                     tok != tokenizer.pad_token_id]

            for token_i, token_id in enumerate(token_ids[instance_i]):

                if cls_aggregation == 'pred_cls':
                    cls_ = pred_classes[instance_i]
                    token_sal = class_attr_list[cls_][instance_i][token_i]
                elif cls_aggregation == 'true_cls':
                    cls_ = instance['label_id']
                    token_sal = class_attr_list[cls_][instance_i][token_i]
                elif cls_aggregation == 'sum':
                    try:
                        token_sal = sum(
                            [class_attr_list[cls_][instance_i][token_i] for cls_ in
                             range(labels)])
                    except:
                        print(class_attr_list[cls_], class_attr_list[cls_][instance_i], class_attr_list[cls_][instance_i][token_i])
                        raise ValueError
                elif cls_aggregation == 'max':
                    token_sal = max(
                        [class_attr_list[cls_][instance_i][token_i] for cls_ in
                         range(labels)])
                word_json['word_scores'].append([token_id, tokenizer.decode([token_id]), token_sal])

                # when reaching a new sentence, first aggregate saliency for the
                # previous
                if token_i > 0 and (token_id == tokenizer.cls_token_id or
                                    token_i == len(token_ids[instance_i]) - 1):
                    # we don't need the saliency for the prediction task
                    if token_i == len(token_ids[instance_i]) - 1:
                        current_sent_sal.append(token_sal)

                    if sentence_aggregation == 'CLS':
                        saliencies.append(current_sent_sal[0])
                    elif sentence_aggregation == 'sum':
                        saliencies.append(sum(current_sent_sal))
                    elif sentence_aggregation == 'max':
                        saliencies.append(max(current_sent_sal))
                    current_sent_sal = []
                current_sent_sal.append(token_sal)

            # remove first sentence which is the question-answer part
            saliencies = saliencies[1:]

            for sent, sent_score in zip(instance['text'], saliencies):
                sent_json['sentence_scores'].append([sent, sent_score])

            # serialize
            out_s.write(json.dumps(sent_json)+'\n')
            out_w.write(json.dumps(word_json)+'\n')

            # compute ROUGE sentence score
            best_sentences = np.argsort(saliencies)[::-1][:n_sentences]
            sentence_pred = [instance['text'][s_id] for s_id in range(len(saliencies)) if s_id in best_sentences ]
            score = scorer.score(prediction='\n'.join(sentence_pred),
                                 target='\n'.join(instance['explanation_text']))
            sent_scores.append(score)

            # compute ROUGE word score
            sorted_words = sorted(word_json['word_scores'], key=lambda x: x[-1], reverse=True)
            sorted_words = [w[1] for w in sorted_words[:60]]

            score = scorer.score(prediction='\n'.join([' '.join(sorted_words[0:15]), ' '.join(sorted_words[15:30]), ' '.join(sorted_words[30:45]), ' '.join(sorted_words[45:])]),
                                 target='\n'.join(instance['explanation_text']))
            word_scores.append(score)

    print('sentence ROUGE:')
    for score_name in score_names:
        print(f'{score_name} P: {np.mean([s[score_name].precision for s in sent_scores]) * 100:.3f} '
              f'R: {np.mean([s[score_name].recall for s in sent_scores]) * 100:.3f} '
              f'F1: {np.mean([s[score_name].fmeasure for s in sent_scores]) * 100:.3f}')


    print('word ROUGE:')
    for score_name in score_names:
        print(f'{score_name} P: {np.mean([s[score_name].precision for s in word_scores]) * 100:.3f} '
              f'R: {np.mean([s[score_name].recall for s in word_scores]) * 100:.3f} '
              f'F1: {np.mean([s[score_name].fmeasure for s in word_scores]) * 100:.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=6)
    parser.add_argument("--dataset", help="Flag for training on gpu",
                        choices=['liar', 'pubhealth'],
                        default='liar')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/e-SNLI/dataset/', type=str)
    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        type=str)
    parser.add_argument("--model_type", help="Type of classification model",
                        choices=['classify',
                                 'extract_classify',
                                 'classify_extract'],
                        default='classify')

    parser.add_argument("--pretrained_path",
                        help="Path where the model will be serialized",
                        default='nli_bert', type=str)
    parser.add_argument("--config_path",
                        help="Path where the model will be serialized",
                        default='nli_bert', type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--accum_steps", help="Gradient accumulation steps",
                        type=int, default=1)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-5)
    parser.add_argument("--max_len", help="Learning Rate", type=int,
                        default=512)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        choices=['train', 'test', 'val'])
    parser.add_argument("--init_only", help="Whether to train the model",
                        action='store_true', default=False)

    parser.add_argument("--n_sentences", help="Number of sentences to select",
                        type=int, default=1)

    parser.add_argument("--token_aggregation",
                        help="Aggregation of the embeddings for a single token",
                        choices=['l2', 'mean'], default='l2')
    parser.add_argument("--cls_aggregation",
                        help="Aggregation of the saliency scores for a single "
                             "token",
                        choices=['pred_cls', 'true_cls', 'sum', 'max'],
                        default='pred_cls')
    parser.add_argument("--sentence_aggregation",
                        help="Aggregation of the saliency scores for a "
                             "sentence",
                        choices=['CLS', 'sum', 'max'],
                        default='CLS')
    parser.add_argument("--test_mode", help="Mode for testing", type=str,
                        default=None, choices=['justification',
                                                               'original_sentences',
                                                               'lead_6',
                                                               'lead_5',
                                                               'lead_4',
                                                               'lead_3',
                                                               'top_5', 'top_6',
                                                               'top_3', 'top_4',
                                                               'selected_sentences',
                                                               'sentences_from_file'])
    parser.add_argument("--sentences_path", help="Path to original split",
                        type=str, default=None)
    parser.add_argument("--file_path", help="Path to final explanations",
                        type=str, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    train, val, test = get_datasets(args, args.dataset_dir,
                                    args.dataset,
                                    args.labels)

    print(f'Train size {len(train)}', flush=True)
    print(f'Dev size {len(val)}', flush=True)

    print('Loaded data...', flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_path)
    pretrained_bert = BertLongForMaskedLM.from_pretrained(
        args.pretrained_path).to(device)
    config = pretrained_bert.config
    transformer_model = pretrained_bert.bert

    config.num_labels = args.labels

    if args.model_type == 'classify':
        model = ClassifierModel(args, transformer_model, config).to(device)
    elif args.model_type == 'extract_classify':
        model = ExtractClassifyJointModel(args, transformer_model, config).to(
            device)

    elif args.model_type == 'classify_extract':
        model = ClassifyExtractJointModel(args, transformer_model, config).to(
            device)

    collate_fn = partial(collate_explanations_joint,
                         tokenizer=tokenizer,
                         device=device,
                         pad_to_max_length=True,
                         max_length=args.max_len,
                         add_logits=False,
                         sep_sentences=True,
                         cls_sentences=True)

    if args.mode == 'test':
        ds = test
    elif args.mode == 'train':
        ds = train
    else:
        ds = val
    dl = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=collate_fn, shuffle=False)
    checkpoint = torch.load(args.model_path)

    model.load_state_dict(checkpoint['model'])

    if args.model_type in ['classify_extract', 'extract_classify']:
        model = ModelWrapper(model)

    eval_sentence_saliency(model,
                       dl,
                       ds,
                       args.token_aggregation,
                       args.cls_aggregation,
                       args.sentence_aggregation,
                       args.labels,
                       n_sentences=args.n_sentences)
