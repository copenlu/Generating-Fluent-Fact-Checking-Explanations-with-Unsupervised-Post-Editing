#!/bin/bash
#Running Pegasus on SA inputs

#############################
echo "LiarPlus Dataset"

echo "1. Running pegasus for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile_filtered liar_sup_val.txt --outdir liar_sup  --device_type gpu --dataset_name liar --outfile_pegasus liar_sup_val_pegasus_onSAinp.txt

echo "1. Pegasus on SA inputs completed for supervised validation justifications!"

#############################

echo "2. Running Pegasus Filtering for supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile_filtered liar_sup_test.txt --outdir liar_sup --device_type gpu --dataset_name liar --outfile_pegasus liar_sup_test_pegasus_onSAinp.txt

echo "2. Pegasus on SA inputs completed for supervised test justifications!"
echo "Completed for Validation/Test of Liar Plus for Supervised"

#############################
#############################

echo "3. Running Pegasus Filtering for Un-supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile_filtered liar_unsup_val.txt --outdir liar_unsup --device_type gpu --dataset_name liar --outfile_pegasus liar_unsup_val_pegasus_onSAinp.txt

echo "3. Pegasus on SA inputs completed for Un-supervised validation justifications!"

#############################

echo "4. Running Pegasus Filtering for Un-supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile_filtered liar_unsup_test.txt --outdir liar_unsup --device_type gpu --dataset_name liar --outfile_pegasus liar_unsup_test_pegasus_onSAinp.txt

echo "4. Pegasus on SA inputs completed for Un-supervised test justifications!"
echo "Completed for Validation/Test of Liar Plus for Un-Supervised"

#############################
#############################

echo "Pubhealth Dataset"

echo "1. Running pegasus for supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile_filtered pub_sup_val.txt --outdir pub_sup  --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_sup_val_pegasus_onSAinp.txt

echo "1. Pegasus on SA inputs completed for supervised validation justifications!"

#############################

echo "2. Running Pegasus Filtering for supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/sup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile_filtered pub_sup_test.txt --outdir pub_sup --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_sup_test_pegasus_onSAinp.txt

echo "2. Pegasus on SA inputs completed for supervised test justifications!"
echo "Completed for Validation/Test of Liar Plus for Supervised"

#############################
#############################

echo "3. Running Pegasus Filtering for Un-supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile_filtered pub_unsup_val.txt --outdir pub_unsup --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_unsup_val_pegasus_onSAinp.txt

echo "3. Pegasus on SA inputs completed for Un-supervised validation justifications!"

#############################

echo "4. Running Pegasus Filtering for Un-supervised test justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile_filtered pub_unsup_test.txt --outdir pub_unsup --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_unsup_test_pegasus_onSAinp.txt

echo "4. Pegasus on SA inputs completed for Un-supervised test justifications!"
echo "Completed for Validation/Test of PubHealth Plus for Supervised"