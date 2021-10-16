import argparse
import math
import random
from argparse import Namespace
from functools import partial
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, \
    accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast
from transformers.optimization import AdamW
from data_loader import get_datasets, collate_explanations_joint
from saliency.pretrain_longformer import BertLongForMaskedLM
from saliency.model_builder import ExtractClassifyJointModel, ClassifyExtractJointModel, ClassifierModel
from rouge_score import rouge_scorer
from multiprocessing import Pool


def train_model(args: Namespace,
                model: torch.nn.Module,
                train_dl: DataLoader, dev_dl: DataLoader,
                optimizer: torch.optim.Optimizer) -> (Dict, Dict):
    best_score, best_model_weights = {'dev_target_f1': 0}, None
    loss_fct = torch.nn.CrossEntropyLoss()
    model.train()
    loss_binary_sentences = torch.nn.BCEWithLogitsLoss().to(device) # pos_weight=torch.tensor([args.pos_sent_loss_weight])

    for ep in range(args.epochs):
        for batch_i, batch in enumerate(train_dl):
            current_step = (step_per_epoch * args.accum_steps) * ep + batch_i

            logits = model(batch=batch)
            if args.model_type != 'classify':
                logits, logits_sentences, one_logit_sentences = logits

            loss = loss_fct(logits.view(-1, args.labels),
                            batch['target_labels_tensor'].long().view(-1)) / args.accum_steps

            loss.backward()

            if (batch_i + 1) % args.accum_steps == 0:
                optimizer.step()  # Now we can do an optimizer step
                # scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()

            current_train = {
                'train_loss': loss.cpu().data.numpy(),
                'epoch': ep,
                'step': current_step,
            }
            print(
                '\t'.join([f'{k}: {v:.3f}' for k, v in current_train.items()]),
                flush=True, end='\r')

            if  ((
                    batch_i % 400 == 0 and batch_i > 0) or batch_i == \
                    step_per_epoch):
                print('\n', flush=True)
                current_val = eval_model(args, model, dev_dl, val)
                current_val.update(current_train)

                print(current_val, flush=True)

                if current_val['dev_target_f1'] > best_score['dev_target_f1']:
                    best_score = current_val
                    best_model_weights = model.state_dict()

                    checkpoint = {
                        'performance': best_score,
                        'args': vars(args),
                        'step': current_step,
                        'model': best_model_weights,
                    }
                    print(f"Saving checkpoint to {args.model_path}",
                          flush=True)
                    torch.save(checkpoint, args.model_path)
                model.train()


        current_val = eval_model(args, model, dev_dl, val)
        current_val.update(current_train)

        print(current_val, flush=True)

        if current_val['dev_target_f1'] > best_score['dev_target_f1']:
            best_score = current_val
            best_model_weights = model.state_dict()

            checkpoint = {
                'performance': best_score,
                'args': vars(args),
                'step': current_step,
                'model': best_model_weights,
            }
            print(f"Saving checkpoint to {args.model_path}",
                  flush=True)
            torch.save(checkpoint, args.model_path)
        model.train()

    return best_model_weights, best_score


def eval_target(args,
                model: torch.nn.Module,
                test_dl: DataLoader):
    model.eval()
    pred_class, true_class, losses, ids = [], [], [], []

    for batch in tqdm(test_dl, desc="Evaluation"):
        optimizer.zero_grad()

        logits = model(batch=batch)
        if args.model_type != 'classify':
            logits = logits[0]

        true_class += batch['target_labels_tensor'].detach().cpu().numpy().tolist()
        pred_class += logits.detach().cpu().numpy().tolist()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, args.labels),
                        batch['target_labels_tensor'].long().view(-1))
        ids += batch['ids']
        losses.append(loss.item())

    prediction_orig = np.argmax(np.asarray(pred_class).reshape(-1, args.labels),
                           axis=-1)

    prediction = prediction_orig

    p, r, f1, _ = precision_recall_fscore_support(true_class,
                                                  prediction,
                                                  average='macro')
    acc = accuracy_score(true_class, prediction)

    return prediction_orig, ids, np.mean(losses), acc, p, r, f1


def eval_model(args,
               model: torch.nn.Module,
               test_dl: DataLoader,
               test):
    prediction, ids, losses, acc, p, r, f1 = eval_target(args, model, test_dl)
    dev_eval = {
        'loss_dev': np.mean(losses),
        'dev_target_p': p,
        'dev_target_recall': r,
        'dev_target_f1': f1,
        'dev_acc': acc
    }

    return dev_eval


def eval_sentences(args,
               model: torch.nn.Module,
               test_dl: DataLoader,
               ds):
    model.eval()

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    top_n_sentences, scores = [], []
    instance_i = 0

    for batch in tqdm(test_dl, desc="Evaluation"):
        optimizer.zero_grad()

        logits = model(batch=batch)
        sentence_logits = logits[-1].detach()

        sentence_logits = sentence_logits * batch['sentences_mask'] + (~batch['sentences_mask'] * (-5e10))

        best_sentences = torch.argsort(sentence_logits, dim=1, descending=True)[:, :args.top_n].cpu().numpy().tolist()
        for instance in best_sentences:
            top_n_sentences.append('\n'.join([ds.dataset[instance_i]['text'][sentence_i] for sentence_i in instance]))
            instance_i += 1

    justification = ['\n'.join(instance['explanation_text'])
                     for instance in ds]

    p = Pool()
    scores = p.starmap(scorer.score, [(pred, just) for pred, just in zip(top_n_sentences, justification)])

    for score_name in score_names:
        print(f'{score_name} P: {np.mean([s[score_name].precision for s in scores]) * 100:.3f} '
              f'R: {np.mean([s[score_name].recall for s in scores]) * 100:.3f} '
              f'F1: {np.mean([s[score_name].fmeasure for s in scores]) * 100:.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=6)
    parser.add_argument("--dataset", help="Name of the dataset",
                        choices=['liar', 'pubhealth'],
                        default='liar')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/liar/', type=str)
    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        type=str)
    parser.add_argument("--pretrained_path",
                        help="Name of the pretrained LM",
                        type=str)
    parser.add_argument("--model_type", help="Type of classification model",
                        choices=['classify',
                                 'extract_classify',
                                 'classify_extract'],
                        default='classify')

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--accum_steps", help="Gradient accumulation steps",
                        type=int, default=1)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-5)
    parser.add_argument("--max_len", help="Learning Rate", type=int,
                        default=512)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        default='train', choices=['train',
                                                  'test',
                                                  'test_dev',
                                                  'test_sentences',
                                                  'dev_sentences'])
    parser.add_argument("--test_mode", help="Mode for testing", type=str,
                        default='original_sentences', choices=['justification',
                                                  'original_sentences',
                                                  'lead_6',
                                                  'lead_5',
                                                  'lead_4',
                                                  'lead_3',
                                                  'top_5', 'top_6', 'top_3', 'top_4',
                                                  'selected_sentences',
                                                  'sentences_from_file'])
    parser.add_argument("--sentences_path", help="Path to original split",
                        type=str)
    parser.add_argument("--file_path", help="Path to final explanations",
                        type=str, default=None)

    parser.add_argument("--init_only", help="Whether to train the model",
                        action='store_true', default=False)
    parser.add_argument("--top_n", help="Eval top n sentences", type=int,
                        default=6)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_path)
    pretrained_bert = BertLongForMaskedLM.from_pretrained(args.pretrained_path).to(device)
    config = pretrained_bert.config
    transformer_model = pretrained_bert.bert

    config.num_labels = args.labels

    if args.model_type == 'classify':
        model = ClassifierModel(args, transformer_model, config).to(device)
    elif args.model_type == 'extract_classify':
        model = ExtractClassifyJointModel(args, transformer_model, config).to(device)
    elif args.model_type == 'classify_extract':
        model = ClassifyExtractJointModel(args, transformer_model, config).to(device)

    collate_fn = partial(collate_explanations_joint,
                         tokenizer=tokenizer,
                         device=device,
                         pad_to_max_length=True,
                         max_length=args.max_len,
                         add_logits=False,
                         sep_sentences=True,
                         cls_sentences=True,
                         sentence_source=args.test_mode)

    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      betas=(0.9, 0.98))

    if args.mode in ['test', 'test_dev']:
        from data_loader import LIARDataset, PubHealth

        if args.dataset == 'liar':
            if args.mode == 'test':
                dataset_path = 'ruling_oracles_test.tsv'
            else:
                dataset_path = 'ruling_oracles_val.tsv'

            ds = LIARDataset(args, f'{args.dataset_dir}/{dataset_path}',
                             num_labels=args.labels)

        elif args.dataset == 'pubhealth':
            if args.mode == 'test':
                dataset_path = 'test.tsv'
            else:
                dataset_path = 'dev.tsv'

            ds = PubHealth(args, f'{args.dataset_dir}/{dataset_path}')

        test_dl = DataLoader(ds, batch_size=args.batch_size,
                             collate_fn=collate_fn, shuffle=False)

        checkpoint = torch.load(args.model_path)

        model.load_state_dict(checkpoint['model'])
        result = eval_model(args, model, test_dl, ds)
        print(result, flush=True)

    else:
        train, val, test = get_datasets(args, args.dataset_dir,
                                        args.dataset,
                                        args.labels,
                                        add_logits=False)
        print(f'Train size {len(train)}', flush=True)
        print(f'Dev size {len(val)}', flush=True)

        print('Loaded data...', flush=True)

        train_dl = DataLoader(batch_size=args.batch_size,
                              dataset=train, shuffle=True,
                              collate_fn=collate_fn)
        dev_dl = DataLoader(batch_size=1,
                            dataset=val,
                            collate_fn=collate_fn,
                            shuffle=False)

        step_per_epoch = math.ceil(len(train) / (args.batch_size * args.accum_steps))
        num_steps = step_per_epoch * args.epochs

        if args.init_only:
            best_model_w, best_perf = model.state_dict(), {
                'loss_dev': 0
            }
        else:
            best_model_w, best_perf = train_model(args, model,
                                                  train_dl, dev_dl,
                                                  optimizer)

        print(best_perf)
        print(args)