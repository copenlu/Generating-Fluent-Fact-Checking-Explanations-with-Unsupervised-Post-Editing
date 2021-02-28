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

        return keyword_embeds.bmm(ref_embeds.permute(0, 2, 1)).max(dim=2).values.min(dim=1).values

    def sentence_level_semantic_scorer(self, new_justs, org_justs):

        new_justs_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(new_justs)
        org_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(org_justs)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(new_justs_embeds, org_embeds)

        return output

    def length_penality(self, justifications):

        len_penalities = []
        for just in justifications:
            len_penalities.append(1/(len(sum(just, []))))

        return len_penalities

    def power(self, my_list, weight):
        return [x**weight for x in my_list]

    def scorer(self, new_justs, original_justs, positions):
        # TODO: optimise to compute the scores for the edited sentence only?

        new_justs_text = [' '.join(sum(sentences, [])) for sentences in new_justs]
        original_justs_text = [' '.join(sum(sentences, [])) for sentences in original_justs]

        fluency_scores = self.gpt_scorer(new_justs_text)
        semantic_scores = self.word_level_semantic_scorer(new_justs_text, original_justs_text)
        length_score = self.length_penality(new_justs)

        # TODO pass in one batch
        nli_score = [
            self.nli_scorer(' '.join(original_justs[i][idx[0]]),
                            ' '.join(new_justs[i][idx[0]]))
                     for i, idx in enumerate(positions)]
        weighted_nli = self.power(nli_score, self.args.nli_weight)

        weighted_length_scores = self.power(length_score, self.args.length_weight)
        sentence_semantic_scores = self.sentence_level_semantic_scorer(new_justs_text, original_justs_text)

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
        'scored_sentences':  'scored_sentences': [['U.S.', 'Rep.', 'Vicky', 'Hartzler', ',', 'R-Columbia', ',', 'said', ',', '``', 'We', 'have', 'less', 'Americans', 'working', 'now', 'than', 'back', 'in', 'the', '‘', '70s', ',', 'so', 'we', 'do', 'need', 'to', 'still', 'address', 'that', 'and', 'get', 'our', 'economy', 'going', '.', "''"], ['This', 'chart', 'shows', 'the', 'ratio', 'throughout', 'the', '1970s', '.'], ['In', '2015', ',', 'that', 'number', 'had', 'reached', 'more', 'than', '155', 'million', '.'], ['The', 'data', 'show', 'that', 'in', 'only', 'one', 'year', ',', '1979', ',', 'the', 'employment-population', 'ratio', 'was', 'higher', 'than', 'it', 'was', 'in', '2015', '.'], ['The', 'rate', 'in', '1970', 'was', '60.4', 'percent', 'and', 'it', 'grew', 'steadily', 'to', '62.3', 'percent', 'in', '1977', ',', 'but', 'it', 'didn', '’', 't', 'rise', 'above', '2015', '’', 's', 'rate', 'in', 'that', 'time', '.'], ['These', 'data', 'show', 'a', 'nearly', '50', 'percent', 'increase', 'in', 'the', 'absolute', 'number', 'of', 'Americans', 'working', 'in', '2015', 'compared', 'to', 'the', 'late', '1970s', '.']
        },...]
        :return:
        """
        original_justs = [instance['scored_sentences'] for
            instance in input_batch]  # To keep track of the original input
        pre_edit_justs = copy.deepcopy(original_justs)

        batch_size = len(input_batch)

        for step in range(self.args.max_steps):
            # gives random values of len==batch size between 0 and 3, mapping
            # to operation functions in the editor
            ops = np.random.randint(0, 3, batch_size)
            # positions consist of index of the sentence and the index of the
            # word in the sentence
            sentence_ids = [random.randint(0, len(instance)-1)
                            for instance in pre_edit_justs]
            sentence_words = [random.randint(0, len(instance[sentence_idx])-1)
                              for instance, sentence_idx in zip(pre_edit_justs,
                                                                sentence_ids)]
            positions = [(sent_id, word_id)
                         for sent_id, word_id in zip(sentence_ids,
                                                     sentence_words)]

            edited_justs = self.editor.edit(pre_edit_justs, ops, positions)
            # TODO add marking of the changed content
            print(edited_justs[0])
            T = max(self.args.t_init - self.args.C * step, 0)
            accept_probs = self.acceptance_prob(edited_justs.tolist(),
                                                pre_edit_justs,
                                                original_justs,
                                                T,
                                                positions)

            for idx, accept_prob in enumerate(accept_probs):
                if accept_prob == 1.0:
                    pre_edit_justs[idx] = edited_justs[idx]

        return pre_edit_justs