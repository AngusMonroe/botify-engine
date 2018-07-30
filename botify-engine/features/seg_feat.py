# -*- coding: utf-8 -*-
import utils
from collections import defaultdict as dd
from features import feat
from scipy import sparse as sp
import numpy as np
import lexicon_generator as lg

class segment_struct:
    """data structure for a text segment. Used for role labeling."""

    def __init__(self, text, start, end, label, entity_name, lang = 'zh'):
        self.label, self.entity_name = label, entity_name
        reversed_index = [None for _ in range(len(text))]
        spans = lg.span_tokenize(text, lang = lang)
        self.text = [t[0] for t in spans]
        for j, t in enumerate(spans):
            for i in range(t[1], t[2]):
                reversed_index[i] = j
        self.start = reversed_index[start]
        self.end = reversed_index[end - 1] + 1
        self.sep = lg.get_sep_by_lang(lang)
        self.content = self.text[self.start: self.end]
        self.context = self.text[: self.start] + ['SEP'] + self.text[self.end:]
        self.char_start, self.char_end = start, end

    def get_substring(self, i, j):
        return self.sep.join(self.text[i: j])

    def get_content_len(self):
        return len(self.content)

    def get_context_len(self):
        return len(self.context)

    def get_cc_sep(self):
        return 'SEP'

class segment_feat_extractor(feat.feat_extractor):

    def fit_extract(self, segment_insts):
        """segment_insts: list of segment_struct
        return: list of dict {feat name: value}.
        """
        feat_dicts = [{} for _ in range(len(segment_insts))]
        for feat in self.feats:
            feat.fit_extract(feat_dicts, segment_insts)
        return feat_dicts

    def extract(self, segment_insts):
        feat_dicts = [{} for _ in range(len(segment_insts))]
        for feat in self.feats:
            feat.extract(feat_dicts, segment_insts)
        return feat_dicts

    def fit_get_tensors(self, feat_dicts, segment_insts, th = 2):
        feat2index, feat_size = {}, 0
        for feat_dict in feat_dicts:
            for k in feat_dict.keys():
                if k not in feat2index:
                    feat2index[k] = feat_size
                    feat_size += 1
        feat_size = max(feat_size, 1) # to prevent zero-shape feature vectors
        label_size, label2index = 0, {}
        content_len, context_len = 0, 0
        for segment_inst in segment_insts:
            if segment_inst.label not in label2index:
                label2index[segment_inst.label] = label_size
                label_size += 1
            content_len = max(content_len, segment_inst.get_content_len())
            context_len = max(context_len, segment_inst.get_context_len())
        self.packed_tensor_data = (feat_size, feat2index, label_size, label2index, content_len, context_len)

        word2cnt = dd(int)
        for segment_inst in segment_insts:
            for word in segment_inst.text:
                word = lg.process_word(word)
                word2cnt[word] += 1
        self.index2word, self.word2index = ['UNK', segment_insts[0].get_cc_sep()], {}
        for word, cnt in word2cnt.items():
            if cnt >= th:
                self.index2word.append(word)
        for i, word in enumerate(self.index2word):
            self.word2index[word] = i

        return self.get_tensors(feat_dicts, segment_insts)

    def get_tensors(self, feat_dicts, segment_insts):
        n = len(feat_dicts)
        feat_size, feat2index, label_size, label2index, content_len, context_len = self.packed_tensor_data
        data, row, col = [], [], []
        for i, feat_dict in enumerate(feat_dicts):
            for k, v in feat_dict.items():
                if k not in feat2index: continue
                data.append(v)
                row.append(i)
                col.append(feat2index[k])
        x = sp.csr_matrix((data, (row, col)), shape = (n, feat_size), dtype = np.float32)
        y = np.zeros(n, dtype = np.int32)

        content_mask = np.zeros((n, content_len), dtype = np.float32)
        content_vec = np.zeros((n, content_len), dtype = np.int32)
        context_mask = np.zeros((n, context_len), dtype = np.float32)
        context_vec = np.zeros((n, context_len), dtype = np.int32)
        for i, segment_inst in enumerate(segment_insts):
            y[i] = label2index[segment_inst.label]
            for j, word in enumerate(segment_inst.content):
                if j >= content_len: break
                if word not in self.word2index:
                    word = 'UNK'
                content_vec[i, j] = self.word2index[word]
                content_mask[i, j] = 1.0
            for j, word in enumerate(segment_inst.context):
                if j >= context_len: break
                word = lg.process_word(word)
                if word not in self.word2index:
                    word = 'UNK'
                context_vec[i, j] = self.word2index[word]
                context_mask[i, j] = 1.0
        return x, y, content_vec, content_mask, context_vec, context_mask

    def get_feat_size(self):
        return self.packed_tensor_data[0]

    def get_class_cnt(self):
        return self.packed_tensor_data[2]

    def get_default_label(self):
        return self.packed_tensor_data[3].keys()[0]

    def get_label(self, index):
        label2index = self.packed_tensor_data[3]
        for k, v in label2index.items():
            if v == index:
                return k

    def get_all_labels(self):
        label2index = self.packed_tensor_data[3]
        return list(label2index.keys())

    def get_content_len(self):
        return self.packed_tensor_data[4]

    def get_context_len(self):
        return self.packed_tensor_data[5]

    def get_word_cnt(self):
        return len(self.index2word)

class in_set_feat:
    def __init__(self, name, th = -1, min_len = -1):
        """name: str, name of the feature
        s: set, set of entities
        """
        self.name = name
        self.th = th
        self.min_len = min_len
        self.init(name)

    def init(self, name):
        s = lg.load_from_short_file(name, full = True, th = self.th, min_len = self.min_len)

        self.s = set()
        for ent in s:
            self.s.add(ent.lower())

    def sanitize(self):
        del self.s

    def unsanitize(self):
        self.init(self.name)

    def fit_extract(self, feat_dicts, segment_insts):
        self.extract(feat_dicts, segment_insts)

    def extract(self, feat_dicts, segment_insts):
        for i in range(len(feat_dicts)):
            self.extract_(feat_dicts[i], segment_insts[i])

    def extract_(self, feat_dict, segment_inst):
        i = segment_inst
        text, start, end = i.text, i.start, i.end
        if i.get_substring(start, end).lower() in self.s:
            feat_dict[u"{}".format(self.name)] = 1.0

"""deprecated"""
class ngram_config:
    ngram_used = [True, True, False, False, False] # unigram to 5gram, false means not extracting features
    thresholds = [2, 2, 2, 2, 2] # at least x ngrams should exist in the data
    before_windows = [2, 2, 7, 7, 7] # the window size before the current word, 0 if unused
    after_windows = [2, 2, 7, 7, 7] # the window size after the current word, 0 if unused
    inside_used = [True, True, True, True, True] # whether extract inside features: text[t: t + 1]

class ngram_feat:
    def __init__(self, name, config):
        """name: str, name of the feature,
        config: ngram_config
        """
        self.name = name
        self.config = config

    def fit_extract(self, feat_dicts, segment_insts):
        """Note: features must be extracted in a batch mode; i.e. all instances should be passed within one function call.
        """
        ngram_cnts = [dd(int), dd(int), dd(int), dd(int), dd(int)]
        for n in range(5):
            for segment_inst in segment_insts:
                text = segment_inst.text
                for i in range(len(text) - n):
                    ngram_cnts[n][text[i: i + n + 1]] += 1
        self.ngram_cnts = ngram_cnts

        self.extract(feat_dicts, segment_insts)

    def extract(self, feat_dicts, segment_insts):
        for i in range(len(feat_dicts)):
            self.extract_(feat_dicts[i], segment_insts[i])

    def extract__(self, feat_dict, segment_inst):
        i = segment_inst
        text, start, end = i.text, i.start, i.end
        for n in range(5):
            if not self.config.ngram_used[n]: continue
            for i in range(start, end):
                if i + n >= end: continue
                if self.ngram_cnts[n][text[i: i + n + 1]] >= self.config.thresholds[n]:
                    feat_dict[u'{}_{}_I_{}'.format(self.name, n, text[i: i + n + 1])] = 1.0
            window_size = self.config.before_windows[n]
            for i in range(start - window_size, start):
                if i < 0 or i + n >= start: continue
                if self.ngram_cnts[n][text[i: i + n + 1]] >= self.config.thresholds[n]:
                    feat_dict[u'{}_{}_B_{}'.format(self.name, n, text[i: i + n + 1])] = 1.0
            window_size = self.config.after_windows[n]
            for i in range(end, end + window_size):
                if i + n >= end + window_size or i + n >= len(text): continue
                if self.ngram_cnts[n][text[i: i + n + 1]] >= self.config.thresholds[n]:
                    feat_dict[u'{}_{}_A_{}'.format(self.name, n, text[i: i + n + 1])] = 1.0

def test():
    text = u"中国的首都是北京市"
    data = []
    data.append(segment_struct(text, 0, 2, 'i'))
    data.append(segment_struct(text, 0, 3, 'o'))
    data.append(segment_struct(text, 3, 5, 'o'))
    data.append(segment_struct(text, 6, 8, 'i'))
    data.append(segment_struct(text, 6, 9, 'o'))

    feat_dicts = segment_feat_extractor().add(in_set_feat('I', set([u'中国', u'北京']))).fit_extract(data)
    for i, feat_dict in enumerate(feat_dicts):
        if data[i].label == 'i':
            assert(feat_dict == {'I': 1.0})
        else:
            assert(feat_dict == {})
    # feat_dicts = segment_feat_extractor().add(ngram_feat('GRAM', ngram_config())).fit_extract(data)
    # fout = utils.open_file('log.file', 'w')
    # for i, feat_dict in enumerate(feat_dicts):
    #     fout.write(u' '.join(sorted(list(feat_dict.keys()))) + u"\n")
    print('test done.')
