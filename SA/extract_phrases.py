#First start StanfordCoreNLPServer in one of the terminals
#Using: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \\n-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \\n-status_port 9000 -port 9000 -timeout 15000 &

import nltk
import json
from nltk.parse.corenlp import CoreNLPParser

parser = CoreNLPParser('http://localhost:9000', encoding="utf-8")
phrase_tags = ['S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT','QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'SBAR']

def extract_phrases(sent):
    phrases = []
    ordered_phrases = []
    tree = next(parser.raw_parse(sent))
    pos = tree.treepositions()

    for i in range(len(pos) - 1, 1, -1):

        if not isinstance(tree[pos[i]], str):
            if tree[pos[i]].label() in phrase_tags:
                phrases.append(tree[pos[i]].leaves())

    for p in phrases[::-1]: #reversed list to get phrase order as in sentence.
        ordered_phrases.append(" ".join(p)) #list of phrases

    unique_phrases = [x for x in set(ordered_phrases)]

    return list(set(unique_phrases))


if __name__=="__main__":
    sent = "Nearly half of Oregon 's children are poor . '' As far as the U.S. government is concerned , about a quarter of the state ’ s children are poor , not half . According to that report , `` nearly 50 % of children are either poor or low-income . '' In order to qualify as poor , a family must make 100 percent or less of the federal poverty level . It was the line about the percentage of poor children in the state that caught one Oregonian reader ’ s attention . Low income refers to families between 100 and 200 percent of the federal poverty level ."
    unique_ordered_phrases = extract_phrases(sent)
    print(unique_ordered_phrases)

