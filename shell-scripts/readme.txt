1. HE.txt is the output of human_eval.sh
2. We use run_sa_liar.sh run_sa_pubhealth.sh to create human_eval.sh, unsup_exps.sh - The only purpose to do it was to make things fast.
3. find_hyps.sh --> To run different settings of hyperparameters.
4. SA-results- Contains shell scripts and outputs for the first set of experiments to run SA results. We used SA outputs obtained in that phase, for most of the splits. For some, we had to re-run due top bach-size like for Unsup-Liar-Val. We used top-6 for Liar (except Unsup-liar-test) , top-5 for Pubhealth