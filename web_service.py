# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
from gevent import pywsgi

from config import *

import json
# from nltk import sent_tokenize
import string

app = Flask(__name__)


@app.route('/inquiry_extract', methods=['GET', 'POST'])
def inquiry_extract():
    """
    input json:
    {
        "subject": "xxxxxx",  # 邮件标题
        "content": "xxxxxx"   # 邮件正文
    }

    return:
    {
        "code": 0,
        "msg": "success",
        "data": [
            {
                "inquiry": "do you have a physical branch or are you branchless and just an online bank",          # 诉求
                "original_sent": "Do you have a physical branch or are you branchless and just an online bank?",   # 在原始邮件中的原文
                "score": 0.7395214741165381                                                                        # 得分
            }
        ]
    }
    """
    resq_data = json.loads(request.get_data())
    subject = resq_data["subject"].strip()
    content = resq_data["content"].strip()

    for c in string.punctuation:
        subject = subject.replace(c, "")
    subject = subject.lower().strip()
    subject_bert_vec = get_bert_sent_vecs([subject])[0]

    # content = re.sub(r'this|that|them|these|those', "it", content)
    for ds in delete_sents:
        content = content.replace(ds, "")
        content = content.replace(ds.lower(), "")
    content = content.strip().strip(string.punctuation).strip()
    content_in_email = content

    doc = nlp(content)
    # 1. 指代消歧
    content = doc._.coref_resolved
    # 2. 去除人名
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            content = content.replace(ent.text, "PERSON")

    content = content.strip().strip(string.punctuation).strip()
    # sent_tokenize_list_ori = sent_tokenize(content)
    sent_tokenize_list_ori = [str(sent).lower() for sent in nlp(content).sents]

    content_in_email = content_in_email.strip().strip(string.punctuation).strip()
    # sent_tokenize_list_in_email = sent_tokenize(content_in_email)
    sent_tokenize_list_in_email = [str(sent) for sent in nlp(content_in_email).sents]

    # 只提取问句
    sent_indexes = [i for i in range(len(sent_tokenize_list_ori)) if sent_tokenize_list_ori[i].strip()[-1] == "?"]
    sent_tokenize_list = [s.strip() for s in sent_tokenize_list_ori if s.strip()[-1] == "?"]
    if len(sent_tokenize_list) == 0:
        return {'code': 0, 'msg': 'success', 'data': []}

    index_to_delete = []
    # 3. 去除标点
    sent_tokenize_list_ = []
    for i in range(len(sent_tokenize_list)):
        s = sent_tokenize_list[i]
        for c in string.punctuation:
            s = s.replace(c, "")
        if s.strip() != "":
            sent_tokenize_list_.append(s.strip())
        else:
            sent_tokenize_list_.append("ALL_OF_PUNCT")
            index_to_delete.append(i)
    sent_tokenize_list = sent_tokenize_list_

    bert_sent_vecs = get_bert_sent_vecs(sent_tokenize_list)
    # 4. 删除停用句
    for i in range(len(bert_sent_vecs)):
        for ssv in stop_sents_vecs:
            if get_cos_similar(bert_sent_vecs[i], ssv) >= 0.9:
                index_to_delete.append(i)
                break
    bert_sent_vecs = [bert_sent_vecs[i] for i in range(len(bert_sent_vecs)) if i not in index_to_delete]
    sent_list = [sent_tokenize_list[i] for i in range(len(sent_tokenize_list)) if i not in index_to_delete]
    sent_indexes = [sent_indexes[i] for i in range(len(sent_indexes)) if i not in index_to_delete]

    scores = {}
    for i in range(0, len(bert_sent_vecs)):
        if subject == "":
            scores[i] = 0
        else:
            scores[i] = get_cos_similar(bert_sent_vecs[i], subject_bert_vec)

    # 后处理
    ranked_sentences = [(scores[i], s, sent_indexes[i]) for i, s in enumerate(sent_list)]
    ranked_sentences = fill_pre(sent_tokenize_list_ori, ranked_sentences, sent_indexes, bert_sent_vecs,
                                need_pre_sents_vecs)
    non_ques_vecs = [(sent_tokenize_list_ori[i], get_bert_sent_vecs([sent_tokenize_list_ori[i]])[0], i) for i in
                     range(len(sent_tokenize_list_ori)) if i not in sent_indexes]
    none_ques_ranked_sentences = fill_non_ques(non_ques_vecs, subject_bert_vec, subject, declarative_words)
    ranked_sentences = sorted(ranked_sentences + none_ques_ranked_sentences, reverse=True)
    data = gen_res_data(ranked_sentences, sent_tokenize_list_in_email)

    result = {'code': 0, 'msg': 'success', 'data': data}
    return result


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8088), app)
    server.serve_forever()
