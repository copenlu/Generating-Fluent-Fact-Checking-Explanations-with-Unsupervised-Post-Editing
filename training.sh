#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=4000M
# we run on the gpu partition and we allocate 1 titanx gpu
#We expect that our program should not run langer than 1 min
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=200:00:00
#SBATCH -p copenlu --gres=gpu:titanrtx:1

# example training script

# training of Longformer model
#python models/pretrain_lonformer_bert.py --max_len 1536 --model_name bert-base-uncased --master_gpu 0 --gpus 0 1
#python saliency/train_classifier.py --dataset_dir ../just_summ/oracles/ --labels 3 --model_path liar_3l_1e5 --lr 1e-5 --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --accum_steps 3

# evaluating saliency-sorted sentences
#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4


#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4 --token_aggregation l2 --cls_aggregation max --sentence_aggregation max
#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4 --token_aggregation mean --cls_aggregation max --sentence_aggregation max
#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4 --token_aggregation l2 --cls_aggregation mean --sentence_aggregation max
#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4 --token_aggregation l2 --cls_aggregation sum --sentence_aggregation max
#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4 --token_aggregation l2 --cls_aggregation pred_cls --sentence_aggregation max
#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4 --token_aggregation l2 --cls_aggregation max --sentence_aggregation CLS
#python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 4 --token_aggregation l2 --cls_aggregation max --sentence_aggregation max
python saliency/sentence_saliency_scores.py --labels 3 --model_path liar_3l_1e5 --dataset_dir ../just_summ/oracles/ --gpu --pretrained_path ../diagnostic-guided-explanations/tmp/bert-base-uncased-1536/ --max_len 1536 --batch_size 3 --n_sentences 6