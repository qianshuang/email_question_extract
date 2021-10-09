# -*- coding: utf-8 -*-

from common import *
import pandas as pd
import spacy
import neuralcoref
from nltk import sent_tokenize

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型
neuralcoref.add_to_pipe(nlp)

all_cols = ["Category", "Topic", "Incoming email subject", "Incoming email content"]
df_final = pd.DataFrame()

df = pd.read_excel("data/banking_emails.xlsx", engine="openpyxl", sheet_name=None)
for k, v in df.items():
    if k == "summary":
        continue
    v_df = v.loc[:, all_cols]
    df_final = pd.concat([df_final, v_df], ignore_index=True)

df_final = df_final.dropna(subset=["Incoming email content"])  # 81行
print(df_final.head())
print(df_final.shape)

all_qs = []

for index, row in df_final.iterrows():
    Incoming_email_content = row["Incoming email content"]
    if (not pd.isnull(row["Incoming email subject"])) and len(Incoming_email_content) < len(
            row["Incoming email subject"]):
        Incoming_email_content = row["Incoming email subject"]

    acc_questions = []
    sent_tokenize_list = sent_tokenize(Incoming_email_content)
    for i in range(len(sent_tokenize_list)):
        s = sent_tokenize_list[i].strip()
        # 1. 提取问句
        if s[-1] != "?":
            continue
        # 2. 剔除无效问句
        while " help me " in s:
            s = s[s.find(' help me ') + 9:]
        while " please " in s:
            s = s[s.find(" please ") + 8:]

        doc = nlp(s)
        verbs = []
        dobjs = []

        for i_ in range(len(doc)):
            token = doc[i_]
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                verbs.append(token.lemma_)
            if token.pos_ == "NOUN" and token.dep_ == "dobj":
                dobjs.append(token.lemma_)

        if len(verbs) == 1 or len(dobjs) == 1:
            if ("resolve" in verbs and ("issue" in dobjs or "problem" in dobjs)) or \
                    ("answer" in verbs and "question" in dobjs):
                continue
        # 3. 指代消歧
        acc_questions.append(str(i) + ". " + doc._.coref_resolved)

    all_qs.append("\n".join(acc_questions))
df_final["extract_questions"] = all_qs
print(df_final)

df_final['Incoming email content'] = df_final['Incoming email content'].apply(
    lambda x: str(x).replace('\t', ' ').replace('\n', ' '))
df_final['extract_questions'] = df_final['extract_questions'].apply(
    lambda x: str(x).replace('\t', ' ').replace('\n', ' '))

df_final.to_csv('data/email_question_result.txt', sep='\t', index=False)
