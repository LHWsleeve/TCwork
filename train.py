# Copyright 2019-present NAVER Corp.
# Apache License v2.0
# !/usr/bin/env Python
# coding=utf-8
# Wonseok Hwang
# Sep30, 2018
import os, sys, argparse, re, json
import annotate_ws

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine
from tqdm import trange

# import multiprocessing as mp
from Tread_run import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12',
                         'chinese': 'chinese_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS' or args.bert_type == \
            'chinese':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args


def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config


def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['==', '>', '<', '!=']  # do not know why 'OP' required. Hence,
    conn_sql = ['', 'and', 'or']  # 条件咧之间的操作

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    n_conn_tiaojian = len(conn_sql)

    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops, n_conn_tiaojian)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config


def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train'):
    model.train()
    model_bert.train()
    ave_loss, one_acc_num, tot_acc_num, ex_acc_num = 0, 0.0, 0.0, 0.0
    cnt = 0  # count the # of examples
    # Engine for SQL querying.
    # 这里别忘了改，引擎要变成新的
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    pbar = tqdm(range(len(train_loader.dataset)//16))

    for iB, t in enumerate(train_loader):
        # t 是一个完整的tok文件
        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance 源自然语言
        # nlu_t: tokenized nlu  分词的问题
        # sql_i: canonical form of SQL query 查询sql
        # sql_q: full SQL query text. Not used.已删除
        # sql_t: tokenized SQL query 分词的问题 = nlu_t
        # tb   : table
        # hs_t : tokenized headers. Not used.
        # hds :   header

        g_sc, g_sa, g_sop, g_wn, g_wc, g_wo, g_wv, g_sel_num_seq, g_sel_ag_seq, conds = get_g(sql_i)
        # g_sel_num_seq真实sel的个数
        # g_sel_ag_seq 包含一个元组，agg个数，sel实际值，agg实际值（list）
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.

        # 这里提取了语义索引
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            print('索引转值出错')
            continue

        # score
        s_scn, s_sc, s_sa, s_sop, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                                 g_scn=g_sel_num_seq, g_sc=g_sc, g_sa=g_sa,
                                                                 g_wn=g_wn, g_wc=g_wc, g_sop=g_sop, g_wo=g_wo,
                                                                 g_wvi=g_wvi)

        # start = time.time()
        # results = []
        # lenth = len(t)
        # g_wvi_corenlp = []

        # 多进程部分
        '''        
        manager = mp.Manager()
        dict = manager.dict()
        pool = mp.Pool(32)
        for x in range(lenth):

            pool.apply_async(gwvi, (dict, x, conds[x], nlu_t[x]))

        pool.close()
        pool.join()

        for idx in range(lenth):
            g_wvi_corenlp.append(dict[idx])

        end = time.time()
        print('runs %0.2f seconds.' % (end - start))

        '''
        # 单进程部分
        # for x in range(len(conds)):
        #     wv_ann1 = []
        #     cond1 = conds[x]
        #     nlu_1 = nlu_t[x]
        #     for conds11 in cond1:
        #         _wv_ann1 = annotate_ws.annotate(str(conds11[2]))
        #         wv_ann11 = _wv_ann1['gloss']
        #         wv_ann1.append(wv_ann11)
        #
        #     try:
        #         wvi1_corenlp = annotate_ws.check_wv_tok_in_nlu_tok(wv_ann1, nlu_1)
        #         g_wvi_corenlp.append(wvi1_corenlp)
        #     except:
        #         print("gwvi构建失败")
        #         print(nlu_1)
        #         exit()

        loss = Loss_sw_se(s_scn, s_sc, s_sa, s_sop, s_wn, s_wc, s_wo, s_wv,
                          g_sel_num_seq, g_sc, g_sa, g_sop, g_wn, g_wc, g_wo, g_wvi)

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # L = loss.item()
        ave_loss += loss.item()

        pbar.update(len(t))
    return ave_loss / cnt


# return acc, aux_out
#
# def report_detail(hds, nlu,
#                   g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
#                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
#                   cnt_list, current_cnt):
#     cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt
#
#     print(f'cnt = {cnt} / {cnt_tot} ===============================')
#
#     print(f'headers: {hds}')
#     print(f'nlu: {nlu}')
#
#     # print(f's_sc: {s_sc[0]}')
#     # print(f's_sa: {s_sa[0]}')
#     # print(f's_wn: {s_wn[0]}')
#     # print(f's_wc: {s_wc[0]}')
#     # print(f's_wo: {s_wo[0]}')
#     # print(f's_wv: {s_wv[0][0]}')
#     print(f'===============================')
#     print(f'g_sc : {g_sc}')
#     print(f'pr_sc: {pr_sc}')
#     print(f'g_sa : {g_sa}')
#     print(f'pr_sa: {pr_sa}')
#     print(f'g_wn : {g_wn}')
#     print(f'pr_wn: {pr_wn}')
#     print(f'g_wc : {g_wc}')
#     print(f'pr_wc: {pr_wc}')
#     print(f'g_wo : {g_wo}')
#     print(f'pr_wo: {pr_wo}')
#     print(f'g_wv : {g_wv}')
#     # print(f'pr_wvi: {pr_wvi}')
#     print('g_wv_str:', g_wv_str)
#     print('p_wv_str:', pr_wv_str)
#     print(f'g_sql_q:  {g_sql_q}')
#     print(f'pr_sql_q: {pr_sql_q}')
#     print(f'g_ans: {g_ans}')
#     print(f'pr_ans: {pr_ans}')
#     print(f'--------------------------------')
#
#     print(cnt_list)
#
#     print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
#           f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
#           f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
#     print(f'===============================')
#

def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()

    print('g_scn/wc/wn/wo dev/test不监督')
    cnt = 0

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    total = 0
    one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_sop, g_wn, g_wc, g_wo, g_wv, g_sel_num_seq, g_sel_ag_seq, conds = get_g(sql_i)

        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_scn, s_sc, s_sa, s_sop, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)

            # prediction
            score = []
            score.append(s_scn)
            score.append(s_sc)
            score.append(s_sa)
            score.append(s_sop)
            tuple(score)
            pr_sql_i1 = model.gen_query(score, nlu_tt, nlu)

            pr_wn, pr_wc, pr_sop, pr_wo, pr_wvi = pred_sw_se(s_sop, s_wn, s_wc, s_wo, s_wv)
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)  # 映射到字符串

            pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
            pr_sql_i = generate_sql_i(pr_sql_i1, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)

        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.

        # # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        one_err, tot_err = model.check_acc(nlu, pr_sql_i, sql_i)
        one_acc_num += (len(pr_sql_i) - one_err)
        tot_acc_num += (len(pr_sql_i) - tot_err)
        total += len(pr_sql_i)

        # Execution Accuracy
        table_ids = []
        for x in range(len(tb)):
            table_ids.append(tb[x]['id'])

        for sql_gt, sql_pred, tid in zip(sql_i, pr_sql_i, table_ids):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'],
                                          sql_pred['cond_conn_op'])
            except:
                ret_pred = None
            ex_acc_num += (ret_gt == ret_pred)

    return ((one_acc_num / total), (tot_acc_num / total), ex_acc_num / total), results


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


if __name__ == '__main__':
    # mp.set_start_method('spawn')

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = ''
    path_wikisql = os.path.join(path_h, 'data', 'wikisql_tok1')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. Load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)

    ## 4.1.
    # To start from the pre-trained models, un-comment following lines.
    # path_model_bert =
    # path_model =
    # model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

    ## 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert, args.fine_tune)

    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    # used to record best score of each sub-task
    best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv, best_wr = 0, 0, 0, 0, 0, 0, 0, 0
    best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx, best_wr_idx = 0, 0, 0, 0, 0, 0, 0, 0
    best_lf, best_lf_idx = 0.0, 0
    best_ex, best_ex_idx = 0.0, 0
    for epoch in range(args.tepoch):
        print('完整训练次数:', epoch)
        # train
        train_loss = train(train_loader,
                           train_table,
                           model,
                           model_bert,
                           opt,
                           bert_config,
                           tokenizer,
                           args.max_seq_length,
                           args.num_target_layers,
                           args.accumulate_gradients,
                           opt_bert=opt_bert,
                           st_pos=0,
                           path_db=path_wikisql,
                           dset_name='train')

        # check DEV
        with torch.no_grad():
            dev_acc, results_dev = test(dev_loader,
                                        dev_table,
                                        model,
                                        model_bert,
                                        bert_config,
                                        tokenizer,
                                        args.max_seq_length,
                                        args.num_target_layers,
                                        detail=False,
                                        path_db=path_wikisql,
                                        st_pos=0,
                                        dset_name='dev', EG=args.EG)

        # print_result(epoch, tra_acc, 'train')
        # print_result(epoch, dev_acc, 'dev')
        i = epoch

        print(
            'Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f' % (
                dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6],
                dev_acc[0][7]))
        # save the best model
        if dev_acc[1] > best_lf:
            best_lf = dev_acc[1]
            best_lf_idx = i + 1
            # torch.save(model.state_dict(), 'saved_model/best_model')
            state = {'model': model.state_dict()}
            torch.save(state, os.path.join('.', 'saved_model/model_best.pt'))

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join('.', 'saved_model/model_bert_best.pt'))
            epoch_best = epoch
            print(epoch_best)

        if dev_acc[2] > best_ex:
            best_ex = dev_acc[2]
            best_ex_idx = i + 1

        # record the best score of each sub-task
        if True:
            if dev_acc[0][0] > best_sn:
                best_sn = dev_acc[0][0]
                best_sn_idx = i + 1
            if dev_acc[0][1] > best_sc:
                best_sc = dev_acc[0][1]
                best_sc_idx = i + 1
            if dev_acc[0][2] > best_sa:
                best_sa = dev_acc[0][2]
                best_sa_idx = i + 1
            if dev_acc[0][3] > best_wn:
                best_wn = dev_acc[0][3]
                best_wn_idx = i + 1
            if dev_acc[0][4] > best_wc:
                best_wc = dev_acc[0][4]
                best_wc_idx = i + 1
            if dev_acc[0][5] > best_wo:
                best_wo = dev_acc[0][5]
                best_wo_idx = i + 1
            if dev_acc[0][6] > best_wv:
                best_wv = dev_acc[0][6]
                best_wv_idx = i + 1
            if dev_acc[0][7] > best_wr:
                best_wr = dev_acc[0][7]
                best_wr_idx = i + 1
        print('Train loss = %.3f' % train_loss)
        print('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))
        print('Best Logic Form: %.3f at epoch %d' % (best_lf, best_lf_idx))
        print('Best Execution: %.3f at epoch %d' % (best_ex, best_ex_idx))
        if (i + 1) % 10 == 0:
            print('Best val acc: %s\nOn epoch individually %s' % (
                (best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv),
                (best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx)))

    # save results for the official evaluation
    # save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

    #
    # # save best model
    # # Based on Dev Set logical accuracy lx
    # acc_lx_t = acc_dev[-2]
    # if acc_lx_t > acc_lx_t_best:
    #     acc_lx_t_best = acc_lx_t
    #     epoch_best = epoch
    #     # save best model
    #     state = {'model': model.state_dict()}
    #     torch.save(state, os.path.join('.', 'model_best.pt') )
    #
    #     state = {'model_bert': model_bert.state_dict()}
    #     torch.save(state, os.path.join('.', 'model_bert_best.pt'))
    #
    # print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
