#!/bin/bash

#############################
echo "Pubhealth Dataset"
echo "1. Running SA code for supervised validation justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile pub_sup_val.txt --outdir pub_sup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name pubhealth --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "1. Running Pegasus Filtering for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile pub_sup_val.txt --outdir pub_sup --gold_path P_valgold_sup.txt --outfile_filtered pub_sup_val_filter.txt --device_type gpu --dataset_name pubhealth

echo "1. SA+Pegasus Filtering completed for supervised validation justifications!"

#############################

echo "2. Running SA code for supervised test justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile pub_sup_test.txt --outdir pub_sup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name pubhealth --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "2. Running Pegasus Filtering for supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile pub_sup_test.txt --outdir pub_sup --gold_path P_testgold_sup.txt --outfile_filtered pub_sup_test_filter.txt --device_type gpu --dataset_name pubhealth

echo "2. SA+Pegasus Filtering completed for supervised test justifications!"
echo "Completed for Validation/Test of Pubhealth for Supervised"

#############################
#############################
echo "3. Running SA code for Un-supervised validation justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile pub_unsup_val.txt --outdir pub_unsup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name pubhealth --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "3. Running Pegasus Filtering for Un-supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_sccores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile pub_unsup_val.txt --outdir pub_unsup --gold_path P_valgold_unsup.txt --outfile_filtered pub_unsup_val_filter.txt --device_type gpu --dataset_name pubhealth

echo "3. SA+Pegasus Filtering completed for Un-supervised validation justifications!"

#############################

echo "4. Running SA code for Un-supervised test justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile pub_unsup_test.txt --outdir pub_unsup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name pubhealth --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "4. Running Pegasus Filtering for Un-supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile pub_unsup_test.txt --outdir pub_unsup --gold_path P_testgold_unsup.txt --outfile_filtered pub_unsup_test_filter.txt --device_type gpu --dataset_name pubhealth

echo "4. SA+Pegasus Filtering completed for Un-supervised test justifications!"
echo "Completed for Validation/Test of Pubhealth for Un-Supervised"


