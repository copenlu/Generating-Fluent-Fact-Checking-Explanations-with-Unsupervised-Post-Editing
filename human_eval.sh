#!/bin/bash

#############################
#Values copied from other shell scripts to prepare data for HE ready
#############################

echo "LiarPlus Dataset"

echo "1. Running post-processing for supervised validation justifications!"

python sa_pp.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_sup_val.txt --outdir liar_sup --outfile_filtered liar_sup_val_pp.txt --dataset_name liar

echo "1. Running pegasus for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile_filtered liar_sup_val_pp.txt --outdir liar_sup  --device_type gpu --dataset_name liar --outfile_pegasus liar_sup_val_pegasus.txt

echo "1. SA+PP+Pegasus completed for supervised validation justifications!"

#############################
echo "2. Running post-processing for supervised test justifications!"

python sa_pp.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_sup_test.txt --outdir liar_sup --outfile_filtered liar_sup_test_pp.txt --dataset_name liar

echo "2. Running Pegasus Filtering for supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile_filtered liar_sup_test_pp.txt --outdir liar_sup --device_type gpu --dataset_name liar --outfile_pegasus liar_sup_test_pegasus.txt

echo "2. SA+PP+Pegasus Filtering completed for supervised test justifications!"
echo "Completed for Validation/Test of Liar Plus for Supervised"

#############################
#############################

echo "Pubhealth Dataset"

echo "1. Running post-processing for supervised validation justifications!"

python sa_pp.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile pub_sup_val.txt --outdir pub_sup --outfile_filtered pub_sup_val_pp.txt --dataset_name pubhealth

echo "1. Running pegasus for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile_filtered pub_sup_val_pp.txt --outdir pub_sup  --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_sup_val_pegasus.txt

echo "1. SA+PP+Pegasus completed for supervised validation justifications!"

#############################

echo "2. Running post-processing for supervised test justifications!"

python sa_pp.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile pub_sup_test.txt --outdir pub_sup --outfile_filtered pub_sup_test_pp.txt --dataset_name pubhealth

echo "2. Running Pegasus Filtering for supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile_filtered pub_sup_test_pp.txt --outdir pub_sup --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_sup_test_pegasus.txt

echo "2. SA+PP+Pegasus Filtering completed for supervised test justifications!"
echo "Completed for Validation/Test of PubHealth Plus for Supervised"