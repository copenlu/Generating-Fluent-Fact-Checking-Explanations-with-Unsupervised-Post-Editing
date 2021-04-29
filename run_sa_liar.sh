#!/bin/bash

#############################
#echo "Pubhealth Dataset"
#echo "1. Running code for unsupervised validation justs"

#python run_sa.py --sentences_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/unsup_scores_pubhealth/filtered_val.json1 --dataset_path /home/jolly/projects/COPE/DATA-COPE-Project-DIKUServer/PUBHEALTH/dev.tsv --batch_size 5 --device_type gpu --dataset_name pub_health --max_steps 250 --outfile pub_unsup_val.txt --insert_th 1.10 --length_weight 1.35 --fluency_weight 1.5 --semantic_weight_sentences 1.2

#############################
echo "LiarPlus Dataset"
echo "1. Running SA code for supervised validation justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val.txt --outdir liar_sup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "1. Running Pegasus Filtering for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val.txt --outdir liar_sup --gold_path L_valgold_sup.txt --outfile_filtered liar_sup_val_filter.txt --device_type gpu --dataset_name liar

echo "1. SA+Pegasus Filtering completed for supervised validation justifications!"

#############################

echo "2. Running code for supervised test justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_test.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_sup_test.txt --outdir liar_sup --device_type gpu --batch_size 10 --max_steps 220 --dataset_name liar --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "1. Running Pegasus Filtering for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_test.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_sup_test.txt --outdir liar_sup --device_type gpu --gold_path L_testgold_sup.txt --outfile_filtered liar_sup_test_filter.txt --dataset_name liar

echo "1. SA+Pegasus Filtering completed for supervised test justifications!"

echo "Completed for Validation/Test of Liar Plus for Supervised"

#############################

#echo "Started Liar Plus for Un-Supervised"
#
#echo "3. Running code for unsupervised validation justifications!"
#
#python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/sentence_scores_val.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_unsup_val.txt --outdir liar_unsup  --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight  --fluency_weight  --semantic_weight_sentences
#
#echo "4. Running code for unsupervised test justifications!"
#
#python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/sentence_scores_test.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_unsup_test.txt --outdir liar_unsup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight  --fluency_weight  --semantic_weight_sentences
#
#echo "Completed for Validation/Test of Liar Plus for Un-Supervised"
#############################


