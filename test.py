# -*- coding: utf-8 -*-

from common import *
import spacy
from NLP_parser import *

parser = spacy.load('en_core_web_sm')

parse = parser("Donald Trump is the worst president of USA, but Hillary is better than him")
print(findSVAOs(parse))
