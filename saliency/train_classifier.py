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
from transformers import BertConfig, BertModel, BertTokenizerFast
from transformers.optimization import AdamW
from data_loader import collate_explanations, get_datasets
from saliency.pretrain_longformer import BertLongForMaskedLM


class ClassifierModel(torch.nn.Module):
    """Prediction is conditioned on the extracted sentences."""
    def __init__(self, args, transformer_model, transformer_config):
        super().__init__()
        self.args = args
        self.transformer_model = transformer_model

        # pooler
        self.dense = torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size)
        self.activation_tanh = torch.nn.Tanh()

        # classification
        self.dropout = torch.nn.Dropout(transformer_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(transformer_config.hidden_size, transformer_config.num_labels)

        self.softmax_pred = torch.nn.Softmax(dim=-1)

    def encode(self, token_ids):
        return self.transformer_model(token_ids, attention_mask=token_ids!=0)[0]
            # attention_mask=attention_mask,

    def forward(self, batch):
        token_ids = batch['input_ids_tensor']

        hidden_states = self.encode(token_ids)

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation_tanh(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits_pred = self.classifier(pooled_output)

        return logits_pred


def train_model(args: Namespace,
                model: torch.nn.Module,
                train_dl: DataLoader, dev_dl: DataLoader,
                optimizer: torch.optim.Optimizer) -> (Dict, Dict):
    best_score, best_model_weights = {'dev_target_f1': 0}, None
    loss_fct = torch.nn.CrossEntropyLoss()
    model.train()

    for ep in range(args.epochs):
        for batch_i, batch in enumerate(train_dl):
            current_step = (step_per_epoch * args.accum_steps) * ep + batch_i

            logits = model(batch)

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

            if ep > 0 and ((
                    batch_i % 200 == 0 and batch_i > 0) or batch_i == \
                    step_per_epoch):
                print('\n', flush=True)
                loss=None
                batch=None
                logits=None
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
                    print(f"Saving checkpoint to {args.model_path[0]}",
                          flush=True)
                    torch.save(checkpoint, args.model_path[0])
                model.train()

    return best_model_weights, best_score


def eval_target(args,
                model: torch.nn.Module,
                test_dl: DataLoader):
    model.eval()
    pred_class, true_class, losses, ids = [], [], [], []

    for batch in tqdm(test_dl, desc="Evaluation"):
        optimizer.zero_grad()

        mask = batch['input_ids_tensor'] != tokenizer.pad_token_id
        logits = model(batch)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=6)
    parser.add_argument("--dataset", help="Flag for training on gpu",
                        choices=['liar'],
                        default='liar')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/liar/', type=str)
    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        nargs='+',
                        type=str)
    parser.add_argument("--pretrained_path",
                        help="Path where the model will be serialized",
                        type=str)

    parser.add_argument("--config_path",
                        help="Path where the model will be serialized",
                        default='nli_bert', type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--accum_steps", help="Gradient accumulation steps", type=int, default=1)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-5)
    parser.add_argument("--max_len", help="Learning Rate", type=int,
                        default=512)
    parser.add_argument("--window_size", help="Learning Rate", type=int,
                        default=500)
    parser.add_argument("--stride", help="Learning Rate", type=int,
                        default=300)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        default='train', choices=['train', 'test', 'test_dev'])
    parser.add_argument("--init_only", help="Whether to train the model",
                        action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    train, val, test = get_datasets(args.dataset_dir,
                                    args.dataset,
                                    args.labels,
                                    add_logits=False)
    val.dataset=val.dataset[:300]
    print(f'Train size {len(train)}', flush=True)
    print(f'Dev size {len(val)}', flush=True)

    print('Loaded data...', flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_path)
    pretrained_bert = BertLongForMaskedLM.from_pretrained(args.pretrained_path).to(device)
    config = pretrained_bert.config
    transformer_model = pretrained_bert.bert

    config.num_labels = args.labels
    model = ClassifierModel(args, transformer_model, config).to(device)

    collate_fn = partial(collate_explanations,
                         tokenizer=tokenizer,
                         device=device,
                         pad_to_max_length=True,
                         max_length=args.max_len,
                         add_logits=False,
                         sep_sentences=True,
                         cls_sentences=True)

    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      betas=(0.9, 0.98))

    if args.mode in ['test', 'test_dev']:
        if args.mode == 'test':
            test_dl = DataLoader(test, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False)
            ds = test
        else:
            test_dl = DataLoader(val, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False)
            ds = val

        results = []
        for mp in args.model_path:
            checkpoint = torch.load(mp)

            model.load_state_dict(checkpoint['model'])
            result = eval_model(args, model, test_dl, ds)
            results.append(result)
            print(result, flush=True)

        print([f'{k}: {np.mean([result[k] for result in results])} ({np.std([result[k] for result in results])})' for k in results[0].keys()], flush=True)

    else:
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

        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w,
        }
        print(best_perf)
        print(args)

        torch.save(checkpoint, args.model_path[0])
