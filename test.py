# -*- coding: utf-8 -*-

from common import *
import neuralcoref
import spacy

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)
#
# s = "What should I do?"
# doc = nlp(s)
#
# print([token.pos_ for token in doc])
# print([token.dep_ for token in doc])
# print([token for token in doc])

txt = "My sister has a son and she loves him."
doc = nlp(txt)

# print(doc._.coref_clusters)
