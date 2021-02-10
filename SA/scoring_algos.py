import torch
import json
import re
import random
import tqdm
import numpy as np

from rake_nltk import Rake


class SimulatedAnnealing:

    def __init__(self, editor, scorer, t_init, C, fluency_weight, semantic_weight, max_steps):

        self.editor = editor
        self.scorer = scorer
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

    def semantic_scorer(self, refs):

        keyword_embeds = self.editor.get_contextual_word_embeddings(list(self.ref_to_keywords(refs)))
        ref_embeds = self.editor.get_contextual_word_embeddings(refs)

        return keyword_embeds.bmm(ref_embeds.permute(0,2,1)).max(dim=2).values.min(dim=1).values

    def scorer(self, ref_news):

        #batch = self.t5_data_prep.get_batch(mrs, ref_news)
        fluency_scores = self.generator_gpt.scorer_batch(ref_news)
        semantic_scores = self.semantic_scorer(ref_news)
        total_scores = fluency_scores.pow(self.fluency_weight) * semantic_scores.pow(self.semantic_weight)

        return total_scores

    def acceptance_prob(self, ref_hats, ref_olds, T):

        accept_hat = torch.exp(self.scorer(ref_hats) - self.scorer(ref_olds) / T)
        return accept_hat.clamp(0.0, 1.0).squeeze().cpu().detach().numpy().tolist()

    def run(self, input_batch):

        """
        :param input_batch: List[(id1, id2, id3...), (just1, just2, just3....)]
        :return:
        """
        ids = list(input_batch[0])
        ref_orgs = list(input_batch[1])

        ref_olds = list(input_batch[1])
        batch_size =  len(ids)
        for t in range(self.max_steps):
            T = max(self.t_init - self.C * t, 0)
            
            ops = np.random.randint(0, 3, batch_size) #gives random values of len==batch size between 0 and 3
            positions = [random.randint(0,len(i.strip().split(" "))-1) for i in ref_olds]

            ref_hats = self.editor.edit(ref_olds, ops, positions)
            accept_probs = self.acceptance_prob(ref_hats.tolist(), ref_olds, T)

            for idx, accept_prob in enumerate(accept_probs):
                if accept_prob==1.0:
                    ref_olds[idx] = ref_hats[idx]

        return ref_olds