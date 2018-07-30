# -*- coding: utf-8 -*-
import sys
from collections import defaultdict as dd

import numpy as np
from scipy import sparse as sp

import lexicon_generator as lg
import utils
from eval import eval
from features import feat


class tagging_struct:
    """data structure for tagging (e.g., entity recognition)"""

    def __init__(self, text, label, lang = 'zh'):
        """text: unicode str,
        label: list of str. each entry corresponds to a character in text,
        lang: str. zh or en.

        self.text: a list of words (str),
        self.label: a list of labels (str)
        """
        reversed_index = [None for _ in range(len(label))]
        self.spans = spans = lg.span_tokenize(text, lang = lang)
        self.text = [t[0] for t in spans]
        for j, t in enumerate(spans):
            for i in range(t[1], t[2]):
                reversed_index[i] = j
        j = 0
        self.label = ['O' for _ in range(len(spans))]
        self.original_label = label
        while j < len(label):
            cur_parse = eval.parse_label(label[j])
            if cur_parse[0] == 'B':
                k = j + 1
                while k < len(label):
                    suc_parse = eval.parse_label(label[k])
                    if suc_parse[0] == 'I' and suc_parse[1] == cur_parse[1]:
                        k += 1
                    else:
                        break
                t_j, t_k = j, k - 1
                while t_j < len(label) and reversed_index[t_j] is None:
                    t_j += 1
                while t_k > -1 and reversed_index[t_k] is None:
                    t_k -= 1
                if t_j <= t_k:
                    jj, kk = reversed_index[t_j], reversed_index[t_k] + 1
                    self.label[jj] = u'B-{}'.format(cur_parse[1])
                    for i in range(jj + 1, kk):
                        self.label[i] = u'I-{}'.format(cur_parse[1])
                j = k
            else:
                j += 1
        self.sep = lg.get_sep_by_lang(lang)

    def get_len(self):
        return len(self.text)

    def get_substring(self, start, end):
        return self.sep.join(self.text[start: end])

    def after_to_before(self, index):
        return self.spans[index][1], self.spans[index][2]

class tagging_feat_extractor(feat.feat_extractor):

    def __init__(self, test_unk = 'UNK'):
        self.test_unk = test_unk
        super(tagging_feat_extractor, self).__init__()

    def fit_extract(self, tagging_insts):
        """call fit_extract for the first time. call extract from the second time on.
        tagging_insts: list of tagging_struct
        return: list of list of feature dicts
        """
        feat_dicts = []
        for tagging_inst in tagging_insts:
            feat_dicts.append([{} for _ in range(tagging_inst.get_len())])
        for feat in self.feats:
            feat.fit_extract(feat_dicts, tagging_insts)
        return feat_dicts

    def extract(self, tagging_insts):
        feat_dicts = []
        for tagging_inst in tagging_insts:
            feat_dicts.append([{} for _ in range(tagging_inst.get_len())])
        for feat in self.feats:
            feat.extract(feat_dicts, tagging_insts)
        return feat_dicts

    def fit_get_tensors(self, feat_dicts, tagging_insts, th = 2):
        """call fit_get_tensors for training data (used for deployment).
        """
        time_len, feat_size, feat2index = 0, 0, {}
        for feat_dict_entry in feat_dicts:
            time_len = max(len(feat_dict_entry), time_len)
            for feat_dict in feat_dict_entry:
                for k in feat_dict.keys():
                    if k not in feat2index:
                        feat2index[k] = feat_size
                        feat_size += 1
        label_size, label2index = 0, {}
        for tagging_inst in tagging_insts:
            for label in tagging_inst.label:
                if label not in label2index:
                    label2index[label] = label_size
                    label_size += 1
        self.packed_tensor_data = (time_len, feat_size, feat2index, label_size, label2index)

        word2cnt = dd(int)
        for tagging_inst in tagging_insts:
            for word in tagging_inst.text:
                word = lg.process_word(word)
                word2cnt[word] += 1
        self.index2word, self.word2index = ['UNK'], {}
        for word, cnt in word2cnt.items():
            if cnt >= th:
                self.index2word.append(word)
        for i, word in enumerate(self.index2word):
            self.word2index[word] = i

        # print 'time_len', time_len, 'feat_size', feat_size, 'label_size', label_size
        print('time_len', time_len, 'feat_size', feat_size, 'label_size', label_size, 'word_cnt', len(self.index2word))
        return self.get_tensors(feat_dicts, tagging_insts)

    def get_tensors(self, feat_dicts, tagging_insts):
        """call get_tensors for test data.
        """
        time_len, feat_size, feat2index, label_size, label2index = self.packed_tensor_data
        index2label = {index: label for label, index in label2index.items()}
        n = len(feat_dicts)
        data, row, col = [], [], []
        mask = np.zeros((n, time_len), dtype = np.float32)
        for i, feat_dict_entry in enumerate(feat_dicts):
            for j, feat_dict in enumerate(feat_dict_entry):
                if j >= time_len: break
                mask[i, j] = 1.0
                for k, v in feat_dict.items():
                    if k not in feat2index: continue
                    data.append(v)
                    row.append(i * time_len + j)
                    col.append(feat2index[k])
        x = sp.csr_matrix((data, (row, col)), shape = (n * time_len, feat_size), dtype = np.float32)
        y = np.zeros((n, time_len), dtype = np.int32)
        word_vec = np.zeros((n, time_len), dtype = np.int32)
        for i, tagging_inst in enumerate(tagging_insts):
            for j, label in enumerate(tagging_inst.label):
                if j >= time_len: break
                y[i, j] = label2index[label]
            for j, word in enumerate(tagging_inst.text):
                if j >= time_len: break
                word = lg.process_word(word)
                # debug univ
                # if u'university_B' in feat_dicts[i][j]:
                #     if y[i, j] != label2index[u'B-大学'] and y[i, j] == label2index[u'B-省份']:
                #         print 'bug'
                #         print word
                #         print index2label[y[i,j]]
                #         print ''.join(tagging_insts[i].text)
                #         print feat_dicts[i][j]
                        # print ' '.join(tagging_insts[i].original_label)
                        # print ' '.join(tagging_insts[i].label)
                        # sys.exit(1)
                if word not in self.word2index:
                    if len(feat_dicts[i][j]) == 0:
                        word = self.test_unk
                    else:
                        word = 'UNK'
                word_vec[i, j] = self.word2index[word]

        # return x, y, mask
        return x, y, mask, word_vec

    def get_word_cnt(self):
        return len(self.index2word)

    def get_time_len(self):
        return self.packed_tensor_data[0]

    def get_feat_size(self):
        return self.packed_tensor_data[1]

    def get_class_cnt(self):
        return self.packed_tensor_data[3]

    def get_index2label(self):
        label2index = self.packed_tensor_data[4]
        index2label = {}
        for label, index in label2index.items():
            index2label[index] = label
        return index2label

    def get_default_label(self):
        """default label is used to construct instances at test time where labels are unknown. Simply a placeholder, which could be improved.
        """
        return self.packed_tensor_data[4].keys()[0]

    def get_label(self, index):
        """returns the label given the index.
        """
        label2index = self.packed_tensor_data[4]
        for k, v in label2index.items():
            if v == index:
                return k

    def get_index2feat(self):
        feat2index = self.packed_tensor_data[2]
        index2feat = {}
        for feat, index in feat2index.items():
            index2feat[index] = feat
        return index2feat

class in_set_feat:
    """BIOES/BIO feature tagging for the given sentence and the reference set"""

    def __init__(self, name, scheme = 'BIOES', th = -1, min_len = -1):
        """name: str, name of the feature in the dict
        s: set, the reference set
        scheme: BIOES or BIO
        """
        self.name = name
        self.scheme = scheme
        self.th = th
        self.min_len = min_len
        self.init(name, scheme)

    def init(self, name, scheme):
        s = lg.load_from_short_file(name, full = True, th = self.th, min_len = self.min_len)

        self.s = set()
        for ent in s:
            self.s.add(ent.lower())

    def sanitize(self):
        del self.s

    def unsanitize(self):
        self.init(self.name, self.scheme)

    def fit_extract(self, feat_dicts, tagging_insts):
        """feat_dicts: list of list of dicts, dict: key = feature name, value = feature value
        tagging_insts: list of tagging_struct
        """
        self.extract(feat_dicts, tagging_insts)

    def extract(self, feat_dicts, tagging_insts):
        for i in range(len(feat_dicts)):
            self.extract_(feat_dicts[i], tagging_insts[i])

    def extract_(self, feat_dict, tagging_inst):
        n = tagging_inst.get_len()
        text = tagging_inst.text
        visited = [False for _ in range(n)]
        for len in reversed(range(1, n + 1)):
            for i in range(n):
                j = i + len
                if j > n: break
                if tagging_inst.get_substring(i, j).lower() in self.s:
                    any_visited = False
                    for k in range(i, j):
                        if visited[k]:
                            any_visited = True
                            break
                    if not any_visited:
                        for k in range(i, j):
                            visited[k] = True
                        if self.scheme == 'BIO':
                            feat_dict[i][u"{}_B".format(self.name)] = 1.0
                            for k in range(i + 1, j):
                                feat_dict[k][u"{}_I".format(self.name)] = 1.0
                        elif self.scheme == 'BIOES':
                            if j == i + 1:
                                feat_dict[i][u"{}_S".format(self.name)] = 1.0
                            else:
                                feat_dict[i][u"{}_B".format(self.name)] = 1.0
                                feat_dict[j - 1][u"{}_E".format(self.name)] = 1.0
                                for k in range(i + 1, j - 1):
                                    feat_dict[k][u"{}_I".format(self.name)] = 1.0
                        else:
                            utils.raise_error('unknown tagging shceme for in_set_feat: {}'.format(self.scheme))

"""deprecated"""
class ngram_config:
    ngram_used = [True, True, False, False, False] # unigram to 5gram, false means not extracting features
    thresholds = [2, 2, 2, 2, 2] # at least x ngrams should exist in the data
    before_windows = [7, 7, 7, 7, 7] # the window size before the current word, 0 if unused
    after_windows = [7, 7, 7, 7, 7] # the window size after the current word, 0 if unused
    inside_used = [True, True, True, True, True] # whether extract inside features: text[t: t + 1]

class ngram_feat:
    """extract frequent ngrams as features, including before, inside and after the current window."""
    def __init__(self, name, config, lang = 'zh'):
        """name: str, feature name,
        config: ngram_config
        """
        self.name = name
        self.config = config
        self.sep = lg.get_sep_by_lang(lang)

    def fit_extract(self, feat_dicts, tagging_insts):
        """Note: features must be extracted in a batch mode; i.e. all instances should be passed within one function call.
        """
        ngram_cnts = [dd(int), dd(int), dd(int), dd(int), dd(int)]
        for n in range(5):
            for tagging_inst in tagging_insts:
                text = tagging_inst.text
                for i in range(len(text) - n):
                    ngram_cnts[n][self.sep.join(text[i: i + n + 1])] += 1
        self.ngram_cnts = ngram_cnts

        self.extract(feat_dicts, tagging_insts)

    def extract(self, feat_dicts, tagging_insts):
        for i in range(len(feat_dicts)):
            self.extract_(feat_dicts[i], tagging_insts[i])

    def extract__(self, feat_dict, tagging_inst):
        text = tagging_inst.text
        for n in range(5):
            if not self.config.ngram_used[n]: continue
            if n == 0 and self.config.inside_used[n]:
                for i in range(len(text)):
                    if self.ngram_cnts[n][text[i]] >= self.config.thresholds[n]:
                        feat_dict[i][u'{}_{}_I_{}'.format(self.name, n, text[i])] = 1.0
            for i in range(len(text)):
                window_size = self.config.before_windows[n]
                for j in range(i - window_size, i):
                    if j < 0 or j + n >= i: continue
                    if self.ngram_cnts[n][text[j: j + n + 1]] >= self.config.thresholds[n]:
                        feat_dict[i][u'{}_{}_B_{}'.format(self.name, n, self.sep.join(text[j: j + n + 1]))] = 1.0
                window_size = self.config.after_windows[n]
                for j in range(i + 1, i + window_size):
                    if j + n >= len(text) or j + n > i + window_size: continue
                    if self.ngram_cnts[n][text[j: j + n + 1]] >= self.config.thresholds[n]:
                        feat_dict[i][u'{}_{}_A_{}'.format(self.name, n, self.sep.join(text[j: j + n + 1]))] = 1.0

def test():
    tagging_inst = tagging_struct(u'中国的首都是北京市', ['b', 'i', 'o', 'o', 'o', 'o', 'b', 'i', 'i'])
    print(tagging_inst.text)
    print(tagging_inst.label)
    tagging_insts = [tagging_inst, tagging_inst]
    config = ngram_config()
    config.ngram_used = [True, True, False, False, False]
    config.before_windows = [2, 2, 2, 2, 2]
    config.after_windows = [2, 2, 2, 2, 2]
    config.inside_used = [True, True, True, True, True]
    extractor = tagging_feat_extractor().add(in_set_feat('LOC', set([u'中国', u'北京', u'北京市', u'是北']), scheme = 'BIO'))
    # .add(ngram_feat('GRAM', ngram_config()))
    feat_dicts = extractor.fit_extract(tagging_insts)
    feat_dict = feat_dicts[1]
    assert(feat_dict[0]['LOC_B'] == 1)
    assert(feat_dict[4]['LOC_B'] == 1)
    assert(feat_dict[5]['LOC_I'] == 1)
    # text = tagging_inst.text
    # feat_dict = extractor.extract(tagging_insts)[1]
    # for i, feat_dict_entry in enumerate(feat_dict):
    #     assert(feat_dict_entry[u'GRAM_0_I_{}'.format(text[i])] == 1)
    #     for j in range(max(0, i - 2), i):
    #         assert(feat_dict_entry[u'GRAM_0_B_{}'.format(text[j])] == 1)
    #     for j in range(i + 1, min(i + 3, len(text))):
    #         assert(feat_dict_entry[u'GRAM_0_A_{}'.format(text[j])] == 1)
    #     if i >= 2:
    #         assert(feat_dict_entry[u'GRAM_1_B_{}'.format(text[i - 2: i])] == 1)
    #     if i + 2 < len(text):
    #         assert(feat_dict_entry[u'GRAM_1_A_{}'.format(text[i + 1: i + 3])] == 1)
    print('test done.')

def test_tagging_struct():
    sents = [u'理科男', u'清华计算机']
    labels = [[u"B-科目", u"I-科目", u"B-性别"], [u"B-学校", u"I-学校", u"B-专业", u"I-专业", u"I-专业"]]
    for i in range(len(sents)):
        t = tagging_struct(sents[i], labels[i])
        print(utils.printable(t.text), utils.printable(t.label))

def test_feat_ext():
    sents = [u'理科男', u'清华计算机', u'湖南', u'湖南文科']
    labels = [[u"B-科目", u"I-科目", u"B-性别"], [u"B-学校", u"I-学校", u"B-专业", u"I-专业", u"I-专业"], [u"B-省份", u"I-省份"], [u"B-省份", u"I-省份", u"B-科目", u"I-科目"]]
    # sents = [u'湖南', u'湖南文科546怎样填志愿？']
    # labels = [[u"B-省份", u"I-省份"], [u"B-省份", u"I-省份", u"B-科目", u"I-科目", u"B-分数", u"I-分数", u"I-分数", u"O", u"O", u"O", u"O", u"O", u"O"]]

    tagging_insts = []
    for i in range(len(sents)):
        tagging_insts.append(tagging_struct(sents[i], labels[i]))

    fe = tagging_feat_extractor()
    for name in ['subject', 'province', 'university']:
        fe.add(in_set_feat(name, scheme = 'BIO'))
    feat_dicts = fe.fit_extract(tagging_insts)
    for i, feat_dict in enumerate(feat_dicts):
        for j in range(len(feat_dict)):
            print(tagging_insts[i].text[j], feat_dict[j].keys())

if __name__ == '__main__':
    # test()
    # test_tagging_struct()
    test_feat_ext()
