import torch
import json
import re
import random
import tqdm
import numpy as np

from rake_nltk import Rake

class SimulatedAnnealing:

    def __init__(self, editor, gpt_scorer, t_init, C, fluency_weight, semantic_weight, max_steps):

        self.editor = editor
        self.gpt_scorer = gpt_scorer
        self.t_init = t_init
        self.C = C
        self.fluency_weight = fluency_weight
        self.semantic_weight = semantic_weight
        self.max_steps = max_steps

    def ref_to_keywords(self, refs):

        r = Rake()
        for ref in refs:

            r.extract_keywords_from_text(ref)
            keywords = r.get_ranked_phrases()

            yield " ".join(keywords)

    def word_level_semantic_scorer(self, new_justs, org_justs):

        keyword_embeds = self.editor.get_contextual_word_embeddings(list(self.ref_to_keywords(org_justs)))
        ref_embeds = self.editor.get_contextual_word_embeddings(new_justs)

        return keyword_embeds.bmm(ref_embeds.permute(0,2,1)).max(dim=2).values.min(dim=1).values

    def scorer(self, new_justs, org_justs):

        fluency_scores = self.gpt_scorer(new_justs)
        semantic_scores = self.word_level_semantic_scorer(new_justs, org_justs)
        total_scores = fluency_scores.pow(self.fluency_weight) * semantic_scores.pow(self.semantic_weight)

        return total_scores

    def acceptance_prob(self, justs_hat, old_justs, org_justs, T):

        accept_hat = torch.exp((self.scorer(justs_hat, org_justs) - self.scorer(old_justs, org_justs)) / T)
        return accept_hat.clamp(0.0, 1.0).squeeze().cpu().detach().numpy().tolist()

    def run(self, input_batch):

        """
        :param input_batch: List[(id1, id2, id3...), (just1, just2, just3....)]
        :return:
        """
        ids = list(input_batch[0])
        org_justs = list(input_batch[1])

        old_justs = list(input_batch[1])
        batch_size =  len(ids)
        for t in range(self.max_steps):
            T = max(self.t_init - self.C * t, 0)
            
            ops = np.random.randint(0, 3, batch_size) #gives random values of len==batch size between 0 and 3
            positions = [random.randint(0,len(i.strip().split(" "))-1) for i in old_justs]

            justs_hat = self.editor.edit(old_justs, ops, positions)
            accept_probs = self.acceptance_prob(justs_hat.tolist(), old_justs, org_justs, T)

            for idx, accept_prob in enumerate(accept_probs):
                if accept_prob==0.95:
                    old_justs[idx] = justs_hat[idx]

        return old_justs