
from features import feat
from scipy import sparse as sp
import numpy as np
from collections import defaultdict as dd
import lexicon_generator as lg

class text_struct:

    def __init__(self, text, label, lang = 'zh'):
        """text: unicode str,
        label: str,
        lang: str. zh or en.
        """
        spans = lg.span_tokenize(text, lang = lang)
        self.text = [t[0] for t in spans]
        self.label = label
        self.sep = lg.get_sep_by_lang(lang)

    def get_len(self):
        return len(self.text)

    def get_substring(self, start, end):
        return self.sep.join(self.text[start: end])

class text_feat_extractor(feat.feat_extractor):

    def fit_extract(self, text_insts):
        feat_dicts = [{} for _ in range(len(text_insts))]
        for feat in self.feats:
            feat.fit_extract(feat_dicts, text_insts)
        return feat_dicts

    def extract(self, text_insts):
        feat_dicts = [{} for _ in range(len(text_insts))]
        for feat in self.feats:
            feat.extract(feat_dicts, text_insts)
        return feat_dicts

    def fit_get_tensors(self, feat_dicts, text_insts, th = 2):
        feat2index, feat_size = {}, 0
        for feat_dict in feat_dicts:
            for k in feat_dict.keys():
                if k not in feat2index:
                    feat2index[k] = feat_size
                    feat_size += 1
        time_len, label_size, label2index = 0, 0, {}
        for text_inst in text_insts:
            time_len = max(len(text_inst.text), time_len)
            if text_inst.label not in label2index:
                label2index[text_inst.label] = label_size
                label_size += 1
        self.packed_tensor_data = (time_len, feat_size, feat2index, label_size, label2index)

        word2cnt = dd(int)
        for text_inst in text_insts:
            for word in text_inst.text:
                word = lg.process_word(word)
                word2cnt[word] += 1
        self.index2word, self.word2index = ['UNK'], {}
        for word, cnt in word2cnt.items():
            if cnt >= th:
                self.index2word.append(word)
        for i, word in enumerate(self.index2word):
            self.word2index[word] = i

        print('time_len', time_len, 'feat_size', feat_size, 'label_size', label_size, 'word_cnt', len(self.index2word))
        return self.get_tensors(feat_dicts, text_insts)

    def get_tensors(self, feat_dicts, text_insts):
        n = len(feat_dicts)
        time_len, feat_size, feat2index, label_size, label2index = self.packed_tensor_data
        data, row, col = [], [], []
        for i, feat_dict in enumerate(feat_dicts):
            for k, v in feat_dict.items():
                if k not in feat2index: continue
                data.append(v)
                row.append(i)
                col.append(feat2index[k])
        x = sp.csr_matrix((data, (row, col)), shape = (n, feat_size), dtype = np.float32)
        y = np.zeros(n, dtype = np.int32)
        mask = np.zeros((n, time_len), dtype = np.float32)
        word_vec = np.zeros((n, time_len), dtype = np.int32)
        for i, text_inst in enumerate(text_insts):
            y[i] = label2index[text_inst.label]
            for j, word in enumerate(text_inst.text):
                if j >= time_len: break
                mask[i, j] = 1.0
                word = lg.process_word(word)
                if word not in self.word2index:
                    word = 'UNK'
                word_vec[i, j] = self.word2index[word]

        return x, y, mask, word_vec

    def get_word_cnt(self):
        return len(self.index2word)

    def get_time_len(self):
        return self.packed_tensor_data[0]

    def get_feat_size(self):
        return self.packed_tensor_data[1]

    def get_class_cnt(self):
        return self.packed_tensor_data[3]

    def get_default_label(self):
        return self.packed_tensor_data[4].keys()[0]

    def get_label(self, index):
        label2index = self.packed_tensor_data[4]
        for k, v in label2index.items():
            if v == index:
                return k

class in_set_feat:
    def __init__(self, name, th = -1, min_len = -1):
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

    def fit_extract(self, feat_dicts, text_insts):
        self.extract(feat_dicts, text_insts)

    def extract(self, feat_dicts, text_insts):
        for i in range(len(feat_dicts)):
            self.extract_(feat_dicts[i], text_insts[i])

    def extract_(self, feat_dict, text_inst):
        n = text_inst.get_len()
        text = text_inst.text
        for l in reversed(range(1, n + 1)):
            for i in range(n):
                j = i + l
                if j > n: break
                if text_inst.get_substring(i, j).lower() in self.s:
                    feat_dict[u"{}".format(self.name)] = 1.0
                    return

"""deprecated"""
class ngram_config:
    ngram_used = [True, True, True, False, False]
    thresholds = [10, 10, 10, 2, 2]

class ngram_feat:
    def __init__(self, name, config, lang = 'zh'):
        self.name = name
        self.config = config
        self.sep = lg.get_sep_by_lang(lang)

    def fit_extract(self, feat_dicts, text_insts):
        ngram_cnts = [dd(int), dd(int), dd(int), dd(int), dd(int)]
        for n in range(5):
            for text_inst in text_insts:
                text = text_inst.text
                for i in range(len(text) - n):
                    ngram_cnts[n][text[i: i + n + 1]] += 1
        self.ngram_cnts = ngram_cnts

        self.extract(feat_dicts, text_insts)

    def extract(self, feat_dicts, text_insts):
        for i in range(len(feat_dicts)):
            self.extract_(feat_dicts[i], text_insts[i])

    def extract__(self, feat_dict, text_inst):
        text = text_inst.text
        for n in range(5):
            if not self.config.ngram_used[n]: continue
            for i in range(len(text) - n):
                if self.ngram_cnts[n][text[i: i + n + 1]] >= self.config.thresholds[n]:
                    key = u'{}_{}_{}'.format(self.name, n, text[i: i + n + 1])
                    feat_dict[key] = 1.0
