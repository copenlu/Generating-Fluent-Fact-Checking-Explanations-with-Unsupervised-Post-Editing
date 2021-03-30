import torch
import random
import numpy as np
import copy
import spacy
# import nlp
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

from rake_nltk import Rake


nlp = spacy.load("en_core_web_lg")


class SimulatedAnnealing:

    def __init__(self, editor, gpt_scorer, args, device):

        self.editor = editor
        self.gpt_scorer = gpt_scorer
        self.args = args

        self.sbert = SentenceTransformer('stsb-distilbert-base', device=device)

    def ref_to_keywords(self, refs):

        r = Rake()
        for ref in refs:

            r.extract_keywords_from_text(ref)
            keywords = r.get_ranked_phrases()

            yield " ".join(keywords)

    def word_level_semantic_scorer(self, new_justs, org_justs):

        keyword_embeds = self.editor.get_contextual_word_embeddings(list(self.ref_to_keywords(org_justs)))
        ref_embeds = self.editor.get_contextual_word_embeddings(new_justs)

        return 0.1 * keyword_embeds.bmm(ref_embeds.permute(0, 2, 1)).max(dim=2).values.min(dim=1).values.cpu()

    def sentence_level_semantic_scorer_roberta(self, new_justs, org_justs):

        new_justs_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(new_justs)
        org_embeds = self.editor.get_contextual_word_embeddings_sentencelevel(org_justs)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(new_justs_embeds, org_embeds)

        return output.cpu()

    def sentence_level_semantic_scorer_sbert(self, new_justs, org_justs):

        new_justs_embeds = self.sbert.encode(new_justs)
        org_embeds = self.sbert.encode(org_justs)

        return torch.FloatTensor([util.pytorch_cos_sim(e1, e2) * 10 for e1, e2 in zip(new_justs_embeds, org_embeds)])

    def length_penality(self, justifications):

        len_penalities = []
        for justification in justifications:
            length = len(justification.strip().split(" "))
            len_penalities.append(1000.0/length)

        return len_penalities

    def get_named_entity_score(self, justifications):

        entity_scores = []
        for justification in justifications:
            doc = nlp(justification.strip())
            entity_scores.append(len(doc.ents)+0.01)

        return entity_scores

    def power(self, my_list, weight):
        return [x**weight for x in my_list]

    def scorer(self, new_justs, original_justs):

        fluency_scores = self.gpt_scorer.scorer_batch(new_justs)
        word_semantic_scores = self.word_level_semantic_scorer(new_justs, original_justs)
        length_score = self.length_penality(new_justs)
        entity_score = self.get_named_entity_score(new_justs)
        sentence_semantic_scores = self.sentence_level_semantic_scorer_sbert(new_justs, original_justs)

        weighted_length_scores = self.power(length_score, self.args.length_weight)
        weighted_entity_scores = self.power(entity_score, self.args.named_entity_score_weight)


        # torch.FloatTensor(weighted_nli) * \
        total_scores = fluency_scores.pow(self.args.fluency_weight) * \
                       word_semantic_scores.pow(self.args.semantic_weight_keywords) * \
                       sentence_semantic_scores.pow(self.args.semantic_weight_sentences) * \
                       torch.FloatTensor(weighted_length_scores) * \
                       torch.FloatTensor(weighted_entity_scores)

        # print("Fluency: ", fluency_scores.item(), fluency_scores.pow(self.args.fluency_weight).item())
        # print("Semanctic_wordlevel: ", word_semantic_scores.item(), word_semantic_scores.pow(self.args.semantic_weight_keywords).item())
        # print("Semanctic_sentencelevel: ",sentence_semantic_scores.item(), sentence_semantic_scores.pow(self.args.semantic_weight_sentences).item())
        # print("Length_score: ", length_score, torch.FloatTensor(weighted_length_scores).item())
        # print("Entity_score: ", entity_score, torch.FloatTensor(weighted_entity_scores).item())
        # print("Total_score: ",total_scores.item())

        return total_scores

    def acceptance_prob(self, edited_justs, pre_edit_justs, original_justs, T):
        # TODO save previous scores for optimisation

        # print("Scores of pre-edit sentences")
        last_scores = self.scorer(pre_edit_justs, original_justs)
        # print("----------")
        # print("Scores of edited sentences")
        candidates_scores = self.scorer(edited_justs, original_justs)

        accept_hat = torch.exp((candidates_scores - last_scores) / T)

        return accept_hat.clamp(0.0, 1.0).cpu().detach().numpy().tolist()

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
        #num_steps=0
        for step in tqdm(range(self.args.max_steps)):

            T = max(self.args.t_init - self.args.C * step, 0)
            ops = np.random.randint(0, 3, batch_size)
            # gives random values of len==batch size between 0 and 3, mapping to operation functions in the editor
            # print(ops)
            edited_justs = self.editor.edit(pre_edit_justs, ops)
            # TODO add marking of the changed content

            # print("\n")
            # print(f"Step: {step} | Op: {ops[0]}")
            # print("Pre-edit sentence: ")
            # print(pre_edit_justs[0])
            # print("Edited sentence:")
            # print(edited_justs[0])

            accept_probs = self.acceptance_prob(edited_justs.tolist(),
                                                pre_edit_justs,
                                                original_justs,
                                                T)

            for idx, accept_prob in enumerate(accept_probs):
                #print("Prob:", accept_prob)
                if accept_prob > random.random(): #To check if the accepted probability is greater than random number
                    pre_edit_justs[idx] = edited_justs[idx]
                    #print("Accepted!")
                    #num_accepts += 1
            #print(num_accepts)

        return pre_edit_justs