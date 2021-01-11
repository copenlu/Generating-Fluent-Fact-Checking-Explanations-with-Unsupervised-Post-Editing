#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=4000M
# we run on the gpu partition and we allocate 1 titanx gpu
#We expect that our program should not run langer than 1 min
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=200:00:00
#SBATCH -p copenlu --gres=gpu:titanrtx:1

# example training script
#python models/supervised_training.py --gpu --labels 2 --dataset multirc --dataset_dir data/multirc/ --model_path joint_mrc_pos2_lambda6 --pos_sent_loss_weight 2 --target_lambda 0.6 --eval_every 300 --epochs 10  --lr 1e-5
