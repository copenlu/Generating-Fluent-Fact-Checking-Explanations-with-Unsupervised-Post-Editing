#!/bin/bash

#############################
echo "LiarPlus Dataset"
echo "1. Running SA code for supervised validation justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val.txt --outdir liar_sup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "1. Running Pegasus Filtering for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val.txt --outdir liar_sup --gold_path L_valgold_sup.txt --outfile_filtered liar_sup_val_filter.txt --device_type gpu --dataset_name liar

echo "1. SA+Pegasus Filtering completed for supervised validation justifications!"

#############################

echo "2. Running SA code for supervised test justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_sup_test.txt --outdir liar_sup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "2. Running Pegasus Filtering for supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_sccores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_sup_test.txt --outdir liar_sup --gold_path L_testgold_sup.txt --outfile_filtered liar_sup_test_filter.txt --device_type gpu --dataset_name liar

echo "2. SA+Pegasus Filtering completed for supervised test justifications!"
echo "Completed for Validation/Test of Liar Plus for Supervised"

#############################
#############################

echo "3. Running SA code for Un-supervised validation justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_unsup_val.txt --outdir liar_unsup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "3. Running Pegasus Filtering for Un-supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_sccores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_unsup_val.txt --outdir liar_unsup --gold_path L_valgold_unsup.txt --outfile_filtered liar_unsup_val_filter.txt --device_type gpu --dataset_name liar

echo "3. SA+Pegasus Filtering completed for Un-supervised validation justifications!"

#############################

echo "4. Running SA code for Un-supervised test justifications!"

python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_sccores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_unsup_test.txt --outdir liar_unsup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "4. Running Pegasus Filtering for Un-supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_sccores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_unsup_test.txt --outdir liar_unsup --gold_path L_testgold_unsup.txt --outfile_filtered liar_unsup_test_filter.txt --device_type gpu --dataset_name liar

echo "4. SA+Pegasus Filtering completed for Un-supervised test justifications!"
echo "Completed for Validation/Test of Liar Plus for Un-Supervised"

