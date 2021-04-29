#!/bin/bash

#############################
#echo "Pubhealth Dataset"
#echo "1. Running code for unsupervised validation justs"

#python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/unsup_scores_pubhealth/filtered_val.json1 --dataset_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/PUBHEALTH/dev.tsv --batch_size 5 --device_type gpu --dataset_name pub_health --max_steps 250 --outfile pub_unsup_val.txt --insert_th 1.10 --length_weight 1.35 --fluency_weight 1.5 --semantic_weight_sentences 1.2

#############################
echo "LiarPlus Dataset"
echo "1. Running code for supervised validation justifications!"

echo "1st Run! - Increased Steps"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val1.txt --batch_size 10 --device_type gpu --max_steps 500 --dataset_name liar_plus 

echo "2nd Run! - Increased fluency, deletion, semantic weight"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val2.txt --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar_plus --length_weight 1.35  --fluency_weight 1.5  --semantic_weight_sentences 1.2

echo "3rd Run! - Increased deletion, fluency, semantic weight"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val3.txt --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar_plus --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 

echo "4th Run! - Increased reorder acceptance, increased fluency, semantic weight"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val4.txt --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar_plus --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.92 

echo "5th Run! - Increased reorder acceptance, decreased deletion acceptance, increased fluency, semantic weight"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val5.txt --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar_plus --fluency_weight 1.5 --semantic_weight_sentences 1.3 --reorder_th 0.92 --delete_th 0.99

echo "6th Run! - Kept default parameters"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val6.txt --batch_size 10 --device_type gpu --dataset_name liar_plus

echo "7th Run! - Kept default parameters + GPT-finetuned"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val7.txt --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar_plus --fluencyscorer_model_id /home/jolly/projects/COPE/gpt_ftliar
#############################

echo "8th Run! - increased fluency, semantic weight, reorder acceptance, decreased deletion acceptance + GPT-finetuned"
python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val8.txt --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar_plus --fluencyscorer_model_id /home/jolly/projects/COPE/gpt_ftliar --fluency_weight 1.5 --semantic_weight_sentences 1.3 --reorder_th 0.95 --delete_th 0.97
 



