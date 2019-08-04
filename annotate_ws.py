# coding=gbk
# !/usr/bin/env python3
# docker run --name corenlp -d -p 9000:9000 vzhong/corenlp-server
# Wonseok Hwang. Jan 6 2019, Comment added
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import records
import ujson as json
from stanza.nlp.corenlp import CoreNLPClient
from tqdm import tqdm
import copy
from wikisql.lib.common import count_lines, detokenize
from wikisql.lib.query import Query
from fencibijiao import *
# from HUADONG import *
import time
import multiprocessing as mp
client = None
import jieba
jieba.load_userdict("/home/sleeve/桌面/TCwork_git/data/wikisql_tok1/userdict.txt")


def annotate(sentence, lower=True):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words, gloss, after = [], [], []
    for s in client.annotate(sentence):
        for t in s:
            words.append(t.word) #词级分词（后面要lower）
            gloss.append(t.originalText) # 词级分词
            after.append(t.after) #原问尾，因为是问句，所以是右侧引号
    if lower:
        words = [w.lower() for w in words]
    return {
        'gloss': gloss,
        'words': words,
        'after': after,
    }


def annotate_example(example, table):
    ann = {'table_id': example['table_id']}
    ann['question'] = annotate(example['question'])
    ann['table'] = {
        'header': [annotate(h) for h in table['header']],
    }
    ann['query'] = sql = copy.deepcopy(example['sql'])
    for c in ann['query']['conds']:
        c[-1] = annotate(str(c[-1]))

    q1 = 'SYMSELECT SYMAGG {} SYMCOL {}'.format(Query.agg_ops[sql['agg']], table['header'][sql['sel']])
    q2 = ['SYMCOL {} SYMOP {} SYMCOND {}'.format(table['header'][col], Query.cond_ops[op], detokenize(cond)) for
          col, op, cond in sql['conds']]
    if q2:
        q2 = 'SYMWHERE ' + ' SYMAND '.join(q2) + ' SYMEND'
    else:
        q2 = 'SYMEND'
    inp = 'SYMSYMS {syms} SYMAGGOPS {aggops} SYMCONDOPS {condops} SYMTABLE {table} SYMQUESTION {question} SYMEND'.format(
        syms=' '.join(['SYM' + s for s in Query.syms]),
        table=' '.join(['SYMCOL ' + s for s in table['header']]),
        question=example['question'],
        aggops=' '.join([s for s in Query.agg_ops]),
        condops=' '.join([s for s in Query.cond_ops]),
    )
    ann['seq_input'] = annotate(inp)
    out = '{q1} {q2}'.format(q1=q1, q2=q2) if q2 else q1
    ann['seq_output'] = annotate(out)
    ann['where_output'] = annotate(q2)
    assert 'symend' in ann['seq_output']['words']
    assert 'symend' in ann['where_output']['words']
    return ann


def find_sub_list(sl, l):
    # from stack overflow.
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))
        else:
            continue

    return results


def check_wv_tok_in_nlu_tok(wv_tok1, nlu_t1):
    """
    Jan.2019: Wonseok
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.

    return:
    st_idx of where-value string token in nlu under CoreNLP tokenization scheme.
    """
    g_wvi1_corenlp = []
    nlu_t1_low = [tok.lower() for tok in nlu_t1]
    for i_wn, wv_tok11 in enumerate(wv_tok1):
        wv_tok11_low = [tok.lower() for tok in wv_tok11]
        results = find_sub_list(wv_tok11_low, nlu_t1_low)
        try:
            st_idx, ed_idx = results[0]
            g_wvi1_corenlp.append([st_idx, ed_idx])

        except:
            wv_tok11_low = ''.join(wv_tok11_low)
            # suoyin = suoyin(bert_config, model_bert, tokenizer, nlu_t1, max_seq_length,
            #                 num_out_layers_n=2, num_out_layers_h=768)
            suoyin = Encode(wv_tok11_low, nlu_t1_low)
            g_wvi1_corenlp.append(suoyin)

    return g_wvi1_corenlp


def not_empty(s):
    return s and s.strip()

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def movestopwords(sentence):
    common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    stopwords = stopwordslist('/home/sleeve/桌面/TCwork_git/data/wikisql_tok1/ChineseStopWords.txt')  # 这里加载停用词的路径
    outstr = []
    for word in sentence:
        if word not in stopwords:
            if word != '\t'and'\n':
                try:
                    word = common_used_numerals_tmp[word]
                except:
                    pass
                outstr.append(str(word))
                # outstr += " "
    return outstr

def annotate_example_ws(example, table, fout, cnt, start):
    """
    Jan. 2019: Wonseok
    Annotate only the information that will be used in our model.
    """
    with open(fout, 'a+') as fo:
        ann = {'table_id': example['table_id']}
        _nlu_ann = list(jieba.cut_for_search(example['question']))
        _nlu_ann = movestopwords(_nlu_ann)

        ann['question'] = example['question']
        a = list(filter(not_empty, _nlu_ann))
        ann['question_tok'] = a
        '''
        训练集和验证集需要使用以下代码，测试集不需要
        '''

        # ann['table'] = {
        #     'header': [annotate(h) for h in table['header']],
        # }
        ann['sql'] = example['sql']
        ann['query'] = sql = copy.deepcopy(example['sql'])

        conds1 = ann['sql']['conds']
        wv_ann1 = []
        for conds11 in conds1:
            _wv_ann1 = list(jieba.cut(str(conds11[2]),cut_all=True))
            wv_ann11 = _wv_ann1
            wv_ann1.append(wv_ann11)

            # Check whether wv_ann exsits inside question_tok

        try:
            wvi1_corenlp = check_wv_tok_in_nlu_tok(wv_ann1, ann['question_tok'])
            ann['wvi_corenlp'] = wvi1_corenlp
        except:
            print('gwvi索引出')
            exit()

        fo.write(json.dumps(ann) + '\n')
        end = time.time()
        print('任务 %d runs %0.2f seconds.' % (cnt,(end - start)))

    # return ann


def is_valid_example(e):
    if not all([h['words'] for h in e['table']['header']]):
        return False
    headers = [detokenize(h).lower() for h in e['table']['header']]
    if len(headers) != len(set(headers)):
        return False
    input_vocab = set(e['seq_input']['words'])
    for w in e['seq_output']['words']:
        if w not in input_vocab:
            print('query word "{}" is not in input vocabulary.\n{}'.format(w, e['seq_input']['words']))
            return False
    input_vocab = set(e['question']['words'])
    for col, op, cond in e['query']['conds']:
        for w in cond['words']:
            if w not in input_vocab:
                print('cond word "{}" is not in input vocabulary.\n{}'.format(w, e['question']['words']))
                return False
    return True


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', default='/home/sleeve/桌面/TCwork_git/data/wikisql_tok1', help='data directory')
    parser.add_argument('--dout', default='/home/sleeve/桌面/TCwork_git/data/wikisql_tok1/test_tok', help='output directory')
    parser.add_argument('--split', default='train', help='comma=separated list of splits to process') #'train,dev,test'
    args = parser.parse_args()

    answer_toy = not True
    toy_size = 10

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # for split in ['train', 'dev', 'test']:
    for split in args.split.split(','):
        fsplit = os.path.join(args.din, split) + '.jsonl'
        ftable = os.path.join(args.din, split) + '.tables.jsonl'
        fout = os.path.join(args.dout, split) + '_tok.jsonl'

        print('annotating {}'.format(fsplit))
        with open(fsplit) as fs, open(ftable) as ft, open(fout, 'wt') as fo:
            print('loading tables')

            # ws: Construct table dict with table_id as a key.
            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                d = json.loads(line)
                tables[d['id']] = d
            print('loading examples')
            n_written = 0
            cnt = -1

            pool = mp.Pool(4)
            for line in tqdm(fs, total=count_lines(fsplit)):
                start = time.time()
                cnt += 1
                d = json.loads(line)
                # a = annotate_example(d, tables[d['table_id']])
                pool.apply_async(annotate_example_ws, (d, tables[d['table_id']], fout, cnt, start))
                # a = annotate_example_ws(d, tables[d['table_id']])
                # fo.write(json.dumps(a) + '\n')
                n_written += 1
            pool.close()
            pool.join()

            print('任务结束')


            if answer_toy:
                if cnt > toy_size:
                    break
            print('wrote {} examples'.format(n_written))
