# -*- coding: utf-8 -*-

import spacy

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型

Incoming_email_content = "Dear Sir or Madam,  May I confirm if there is a branch near my hometown? Signal Village Taguig? Thanks, Jes"
Incoming_email_content = Incoming_email_content.lower()
doc = nlp(Incoming_email_content)

for ent in doc.ents:
    print(ent.text, ent.label_)  # Jes PERSON
