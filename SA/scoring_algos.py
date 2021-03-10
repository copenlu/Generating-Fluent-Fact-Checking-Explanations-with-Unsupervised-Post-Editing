import torch
import random
import numpy as np
import copy

from rake_nltk import Rake


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

        return keyword_embeds.bmm(ref_embeds.permute(0, 2, 1)).max(dim=2).values.min(dim=1).values.cpu()

    def sentence_level_semantic_scorer(self, new_justs, org_justs):

        new_justs_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(new_justs)
        org_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(org_justs)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(new_justs_embeds, org_embeds)

        return output.cpu()

    def length_penality(self, justifications):

        len_penalities = []
        for justification in justifications:
            length = len(justification.strip().split(" "))
            len_penalities.append(1.0/length)

        return len_penalities

    def power(self, my_list, weight):
        return [x**weight for x in my_list]

    def scorer(self, new_justs, original_justs, positions):
        # TODO: optimise to compute the scores for the edited sentence only?

        fluency_scores = self.gpt_scorer.scorer_batch(new_justs)
        semantic_scores = self.word_level_semantic_scorer(new_justs, original_justs)
        length_score = self.length_penality(new_justs)

        # TODO pass in one batch
        nli_score = [self.nli_scorer(org, new) for org, new in zip(original_justs, new_justs)]

        weighted_nli = self.power(nli_score, self.args.nli_weight)

        weighted_length_scores = self.power(length_score, self.args.length_weight)
        sentence_semantic_scores = self.sentence_level_semantic_scorer(new_justs, original_justs)

        total_scores = fluency_scores.pow(self.args.fluency_weight) * \
                       torch.FloatTensor(weighted_nli) * \
                       semantic_scores.pow(self.args.semantic_weight) * \
                       sentence_semantic_scores.pow(self.args.semantic_weight) * \
                       torch.FloatTensor(weighted_length_scores)

        return total_scores

    def acceptance_prob(self, edited_justs, pre_edit_justs, original_justs, T, positions):
        # TODO save previous scores for optimisation
        accept_hat = torch.exp((self.scorer(edited_justs, original_justs, positions) -
                                self.scorer(pre_edit_justs, original_justs, positions)) / T)
        return accept_hat.clamp(0.0, 1.0).squeeze().cpu().detach().numpy().tolist()

    def run(self, input_batch):
        """
        :param input_batch: List[{
        'id': '12134.json',
        'statement': 'We have less Americans working now than in the 70s.',
        'justification': 'Hartzler said, ...',
        'label': 'barely-true',
        'scored_sentences': 'U.S. Rep. Vicky Hartzler, R-Columbia, said, .....',
        },...]
        :return:
        """

        original_justs = [instance['scored_sentences'] for instance in input_batch]  # To keep track of the original input
        pre_edit_justs = copy.deepcopy(original_justs)

        batch_size = len(input_batch)

        for step in range(self.args.max_steps):

            T = max(self.args.t_init - self.args.C * step, 0)
            ops = np.random.randint(0, 3, batch_size)
            # gives random values of len==batch size between 0 and 3, mapping to operation functions in the editor


            # positions consist of index of the sentence and the index of the word in the sentence

            positions = [random.randint(0, len(i.strip().split(" ")) - 1) for i in pre_edit_justs]

            edited_justs = self.editor.edit(pre_edit_justs, ops, positions)
            # TODO add marking of the changed content

            accept_probs = self.acceptance_prob(edited_justs.tolist(),
                                                pre_edit_justs,
                                                original_justs,
                                                T,
                                                positions)

            for idx, accept_prob in enumerate(accept_probs):
                if accept_prob == 1.0:
                    pre_edit_justs[idx] = edited_justs[idx]

        return pre_edit_justs