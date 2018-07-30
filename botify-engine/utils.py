import io
from collections import defaultdict as dd
from config import *

import jieba


def raise_error(log_info):
    raise Exception(log_info)

def open_file(filename, mode = 'r'):
    return io.open(filename, mode, encoding = 'utf8')

def get_freq_chat(lang = 'zh'):
    if lang != 'zh':
        print('get_freq_chat not implemented for {}'.format(lang))
        return dd(int)
    freq = dd(int)
    for line in open_file('{}/chat_freq.txt'.format(DATA_DIRECTORY)):
        inputs = line.strip().split(u'\t')
        if len(inputs) < 2: continue
        freq[inputs[0]] = int(inputs[1])
    return freq

def get_freq_general(lang = 'zh'):
    if lang != 'zh':
        print('get_freq_general not implemented for {}'.format(lang))
        return dd(int)
    t = jieba.Tokenizer()
    d = t.gen_pfdict(t.get_dict_file())[0]
    return dd(int, d)

def printable(r):
    return repr(r)


freq_general, freq_chat = get_freq_general(), get_freq_chat()

"""
sort the ambiguous set according to
1. number of characters
2. word length (currently not using this one)
3. frequency of word in a general corpus
4. frequency of word in a chit-chat corpus
"""
def rank_candidate_params(param):
    return [param.start - param.end, -param.pos_count, param.neg_count, freq_general[param.value], freq_chat[param.value]]
    # return [param.start-param.end, -len(param.value), freq_general[param.value], freq_chat[param.value]]
