#!/usr/bin/env python
# -*- coding:utf-8 -*-
# MyThread为自定义的threading.Thread的派生类
# import torch
from annotate_ws import *
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel
# device = torch.device(0)
import os


# def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
#     bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
#     vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
#     init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')
#
#     bert_config = BertConfig.from_json_file(bert_config_file)
#     tokenizer = tokenization.FullTokenizer(
#         vocab_file=vocab_file, do_lower_case=do_lower_case)
#     bert_config.print_status()
#
#     model_bert = BertModel(bert_config)
#     if no_pretraining:
#         pass
#     else:
#         model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
#         print("Load pre-trained parameters.")
#     model_bert.to(device)
#
#     return model_bert, tokenizer, bert_config

def gwvi(dict, x, cond1, nlu_1, max_seq_length, model_bert, bert_config, tokenizer,):

    print('Run task %s...' % (x))

    # model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
    #                                               args.no_pretraining)
    wv_ann1 = []
    for conds11 in cond1:
        _wv_ann1 = annotate(str(conds11[2]))
        wv_ann11 = _wv_ann1['gloss']
        wv_ann1.append(wv_ann11)

    wvi1_corenlp = check_wv_tok_in_nlu_tok(wv_ann1, nlu_1)
    dict[x] = wvi1_corenlp



