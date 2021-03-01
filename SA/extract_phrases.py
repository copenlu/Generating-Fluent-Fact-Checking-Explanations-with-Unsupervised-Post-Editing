#First start StanfordCoreNLPServer in one of the terminals
#Using: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \\n-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \\n-status_port 9000 -port 9000 -timeout 15000 &

import nltk
from nltk.parse.corenlp import CoreNLPParser

parser = CoreNLPParser('http://localhost:9000')
phrase_tags = ['S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT','QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'SBAR']

s =[]
phrases = []
ordered_phrases = []

def extract_phrases(sent):
    tree = next(parser.raw_parse(sent))
    pos = tree.treepositions()

    for i in range(len(pos) - 1, 1, -1):

        if not isinstance(tree[pos[i]], str):
            if tree[pos[i]].label() in phrase_tags:
                phrases.append(tree[pos[i]].leaves())

    for p in phrases[::-1]: #reversed list to get phrase order as in sentence.
        ordered_phrases.append(" ".join(p)) #list of phrases

    unique_phrases = set(ordered_phrases)
    unique_ordered_phrases = [x for x in ordered_phrases if x in unique_phrases]

    print("Phrases:", phrases)
    print("ordered_phrases:", ordered_phrases)
    print("unique_phrases:", unique_phrases)
    print("unique_ordered_phrases:", unique_ordered_phrases)

    return unique_ordered_phrases


if __name__=="__main__":
    sent = "This chart shows the ratio throughout the 1970s. In 2015, that number had reached more than 155 million."
    unique_ordered_phrases = extract_phrases(sent)

