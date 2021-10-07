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
