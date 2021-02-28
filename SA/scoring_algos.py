import torch
import json
import re
import random
import tqdm
import numpy as np

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity


class SimulatedAnnealing:

    def __init__(self, editor, gpt_scorer, nli_scorer, args):

        self.editor = editor
        self.gpt_scorer = gpt_scorer
        self.nli_scorer = nli_scorer
        self.args = args

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

    def sentence_level_semantic_scorer(self, new_justs, org_justs):

        new_justs_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(new_justs)
        org_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(org_justs)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(new_justs_embeds, org_embeds)

        return output

    def length_penality(self, justifications):

        len_penalities = []
        for just in justifications:
            len_penalities.append(1/(len(just.split(" "))))

        return len_penalities

    def power(self, my_list, weight):
        return [x**weight for x in my_list]

    def scorer(self, new_justs, original_justs):

        fluency_scores = self.gpt_scorer(new_justs)
        semantic_scores = self.word_level_semantic_scorer(new_justs, original_justs)
        length_score = self.length_penality(new_justs)
        weighted_length_scores = self.power(length_score, self.args.length_weight)
        sentence_semantic_scores = self.sentence_level_semantic_scorer(new_justs, original_justs)

        total_scores = fluency_scores.pow(self.args.fluency_weight) * semantic_scores.pow(self.args.semantic_weight) * sentence_semantic_scores.pow(self.args.semantic_weight) * torch.FloatTensor(weighted_length_scores)

        return total_scores

    def acceptance_prob(self, edited_justs, pre_edit_justs, original_justs, T):
        # TODO save previous for optimised
        accept_hat = torch.exp((self.scorer(edited_justs, original_justs) - self.scorer(pre_edit_justs, original_justs)) / T)
        return accept_hat.clamp(0.0, 1.0).squeeze().cpu().detach().numpy().tolist()

    def run(self, input_batch):

        """
        :param input_batch: List[(id1, id2, id3...), (just1, just2, just3....)]
        :return:
        """
        ids = list(input_batch[0])
        original_justs = list(input_batch[1]) #To keep track of the original input
        pre_edit_justs = list(input_batch[1])#The justifications on which SA will be applied

        batch_size =  len(ids)
        for t in range(self.args.max_steps):
            T = max(self.args.t_init - self.args.C * t, 0)
            
            ops = np.random.randint(0, 3, batch_size) #gives random values of len==batch size between 0 and 3
            positions = [random.randint(0,len(i.strip().split(" "))-1) for i in pre_edit_justs]

            edited_justs = self.editor.edit(pre_edit_justs, ops, positions)
            print(edited_justs[0])
            accept_probs = self.acceptance_prob(edited_justs.tolist(), pre_edit_justs, original_justs, T)

            for idx, accept_prob in enumerate(accept_probs):
                if accept_prob==1.0:
                    pre_edit_justs[idx] = edited_justs[idx]

        return pre_edit_justs