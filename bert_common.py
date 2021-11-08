# -*- coding: utf-8 -*-

from bert import tokenization
from bert import modeling
import tensorflow as tf

max_seq_length = 128
vocab_file = "export/vocab.txt"
bert_config = modeling.BertConfig.from_json_file("export/bert_config.json")
predict_fn = tf.contrib.predictor.from_saved_model("export")


def convert_single_example(text):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tokens_a = tokenizer.tokenize(text)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids


def get_bert_sent_vecs(sent_list):
    feed_dict = {"input_ids": [], "input_mask": [], "segment_ids": []}
    for line in sent_list:
        input_ids, input_mask, segment_ids = convert_single_example(line.strip())
        feed_dict["input_ids"].append(input_ids)
        feed_dict["input_mask"].append(input_mask)
        feed_dict["segment_ids"].append(segment_ids)
    prediction = predict_fn(feed_dict)
    query_output = prediction["query_output"]
    return query_output
