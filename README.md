# low_resource_fact_checking

# Dependencies can be found in requirements.txt

# Running baselines LEAD-N:
```python3.9 baselines.py --labels 4 --dataset pubhealth --baseline lead --leadn 6 --dataset_dir data/PUBHEALTH```
```python3.9 baselines.py --labels 4 --dataset liar --baseline lead --leadn 6 --dataset_dir data/liar```

# Phrase extraction Set-up:
```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
cd stanford-corenlp-full-2018-10-05
wget http://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-english.jar

nohup java -Djava.io.tmpdir=<local_tmp_dir_here> -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &> tagger.log &
```

# Downloading packages
```
python -m spacy download en_core_web_lg
```

# Running SA
python run_sa.py --sentences_path /image/image-copenlu/unsupervised_fc/sup_sccores/results_serialized_val_filtered.jsonl --dataset_path ../just_summ/oracles/ruling_oracles_val.tsv --length_weight 20 --sample 4 --batch_size 4

