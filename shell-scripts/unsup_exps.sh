#!/bin/bash

#############################
#Values copied from other shell scripts to prepare data other than human evaluation. We are using SA results already stored in iteration 1: (27.04.2021 to 03.05.2021) of google doc.  
#############################

echo "LiarPlus Dataset"

#############################

#echo "3. Running SA code for Un-supervised validation justifications!"

#python run_sa.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_unsup_val.txt --outdir liar_unsup --batch_size 10 --device_type gpu --max_steps 220 --dataset_name liar --length_weight 1.4 --fluency_weight 1.5 --semantic_weight_sentences 1.2 --reorder_th 0.94

echo "3. Running post-processing for Un-supervised validation justifications!"

python sa_pp.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile liar_unsup_val.txt --outdir liar_unsup --outfile_filtered liar_unsup_val_pp.txt --dataset_name liar

echo "3. Running Pegasus Filtering for Un-supervised validation justifications!"

python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_val.tsv --outfile_filtered liar_unsup_val_pp.txt --outdir liar_unsup --device_type gpu --dataset_name liar --outfile_pegasus liar_unsup_val_pegasus.txt

echo "3. SA+PP+Pegasus Filtering completed for Un-supervised validation justifications!"

#############################

#echo "4. Running post-processing for Un-supervised test justifications!"

#python sa_pp.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile liar_unsup_test.txt --outdir liar_unsup --outfile_filtered liar_unsup_test_pp.txt --dataset_name liar

#echo "4. Running Pegasus Filtering for Un-supervised test justifications!"

#python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_liar/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/liar_data/ruling_oracles_test.tsv --outfile_filtered liar_unsup_test_pp.txt --outdir liar_unsup --device_type gpu --dataset_name liar --outfile_pegasus liar_unsup_test_pegasus.txt

#echo "4. SA+PP+Pegasus Filtering completed for Un-supervised test justifications!"
#echo "Completed for Validation/Test of Liar Plus for Un-Supervised"

#############################
#############################

#echo "Pubhealth Dataset"

#echo "3. Running post-processing for Un-supervised validation justifications!"

#python sa_pp.py --sentences_path //home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile pub_unsup_val.txt --outdir pub_unsup --outfile_filtered pub_unsup_val_pp.txt --dataset_name pubhealth

#echo "3. Running Pegasus Filtering for Un-supervised validation justifications!"

#python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_val_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/dev.tsv --outfile_filtered pub_unsup_val_pp.txt --outdir pub_unsup --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_unsup_val_pegasus.txt

#echo "3. SA+PP+Pegasus Filtering completed for Un-supervised validation justifications!"

#############################

# echo "4. Running post-processing for Un-supervised test justifications!"

# python sa_pp.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile pub_unsup_test.txt --outdir pub_unsup --outfile_filtered pub_unsup_test_pp.txt --dataset_name pubhealth

# echo "4. Running Pegasus Filtering for Un-supervised test justifications!"

# python run_pegasus.py --sentences_path /home/jolly/projects/COPE/FilteredData/unsup_scores_pubhealth/results_serialized_test_filtered.jsonl --dataset_path /home/jolly/projects/COPE/FilteredData/PUBHEALTH/test.tsv --outfile_filtered pub_unsup_test_pp.txt --outdir pub_unsup --device_type gpu --dataset_name pubhealth --outfile_pegasus pub_unsup_test_pegasus.txt

# echo "4. SA+PP+Pegasus Filtering completed for Un-supervised test justifications!"
# echo "Completed for Validation/Test of PubHealth Plus for Un-Supervised"
