# -*- coding: utf-8 -*-

"""
1. 提取问句
2. 指代消歧
3. Can you please help me resolve this issue on my mobile app account?
   What should I do?
   could you please answer the following questions?
   I'd also like to open an account with multiple purpose in it. Can you suggest any?
4. 标题与正文相似度去重
5. 对于help me, I need to，提取后面的并与标题计算语义相似度及关键词匹配度，去掉my等
"""

from common import *
import pandas as pd
import spacy
from nltk import sent_tokenize

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型

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

for row in df.iterrows():
    Incoming_email_content = row["Incoming email content"]

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

        for i_ in len(doc):
            token = doc[i]
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                verbs.append(token.lemma_)
            if token.pos_ == "NOUN" and token.dep_ == "dobj":
                dobjs.append(token.lemma_)

        if len(verbs) == 1 or len(dobjs) == 1:
            if ("resolve" in verbs and ("issue" in dobjs or "problem" in dobjs)) or \
                    ("answer" in verbs and "question" in dobjs):
                continue
        # 3. 指代消歧

