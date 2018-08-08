# -*- coding: utf-8 -*-
import utils
from collections import defaultdict as dd
import jieba
from nltk.tokenize import WordPunctTokenizer # Can later be changed to high-end tokenizers
from config import *
import re
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

FULL, PARTIAL, EMPTY = 4, 2, 1

def process_word(word):
    word = re.sub('\d', '0', word)
    word = word.lower()
    lem_word = wordnet_lemmatizer.lemmatize(word, pos = 'n')
    if lem_word == word:
        lem_word = wordnet_lemmatizer.lemmatize(word, pos = 'v')
    word = lem_word
    return word

def decouple(entities, alias, first = True):
    if not first and alias in entities:
        return True
    for i in range(len(alias) - 1):
        if alias[: i] in entities:
            flag = decouple(entities, alias[i:], False)
            if flag:
                return True
    return False

def filter(alias2ent, freq_chat, th_chat, freq_general, th_general, freq_lex, th_lex, filter_set, complex_filter_set, exceptions):
    entities = []
    for filename in filter_set:
        t_entities = load_from_file(filename, full = True)
        entities.extend(t_entities)
    entities = set(entities)

    complex_entities = []
    for filename in complex_filter_set:
        t_entities = load_from_file(filename, full = True)
        complex_entities.extend(t_entities)
    complex_entities = set(complex_entities)

    keys = alias2ent.keys()
    exceptions = set(exceptions)
    fout = utils.open_file('filter.txt', 'w')
    for alias in keys:
        if alias in exceptions: continue
        # if len(alias) <=1 or alias in entities:
        if len(alias) <= 1 or alias in entities or decouple(complex_entities, alias, True):
            alias2ent[alias] = []
            continue
        if freq_chat[alias] > th_chat or freq_general[alias] > th_general or freq_lex[alias] > th_lex:
            alias2ent[alias] = []
            fout.write(u"{}\t{}\t{}\t{}\n".format(alias, freq_chat[alias], freq_general[alias], freq_lex[alias]))
    fout.close()

def comp_term_freq(entities):
    term_freq = dd(int)
    for ent in entities:
        for token in jieba.lcut(ent):
            term_freq[token] += 1
    return term_freq

def merge_tokens(tokens, word, keep_span = False):
    clusters = dd(list)
    for t in tokens:
        clusters[t[1]].append(t)

    sequences = []

    def fun(pre_seg):
        cur_seg = []
        for s in pre_seg:
            cur_seg.append(s)
        # print(cur_seg)
        if cur_seg[-1][2] in clusters:
            for seg in clusters[cur_seg[-1][2]]:
                # print("seg", cur_seg[-1][2], seg)
                fun(cur_seg + [seg])
        elif cur_seg[-1][2] < len(word):
            i = cur_seg[-1][2]
            fun(cur_seg + [(word[i], i, i + 1)])
        else:
            # print("fin", cur_seg)
            sequences.append(cur_seg)
        return cur_seg

    for c in clusters[0]:
        fun([c])

    if len(sequences) == 0:
        return []

    if keep_span:
        return sorted(sequences, key = lambda x: len(x), reverse = True)[0]

    data = []
    for seq in sequences:
        data.append([s[0] for s in seq])

    return sorted(data, key=lambda x: len(x), reverse=True)[0]


def get_partial(word, part_2):
    yield word[0]
    if len(word) == 2 and part_2:
        yield word[1]


def dfs(words, modes, i, res, subs, part_2):
    if i == len(words):
        new_word = u"".join(subs)
        if new_word != u"":
            res.append(new_word)
        return

    if modes[i] & EMPTY:
        subs[i] = u""
        dfs(words, modes, i + 1, res, subs, part_2)
    if modes[i] & PARTIAL:
        for partial in get_partial(words[i], part_2):
            subs[i] = partial
            dfs(words, modes, i + 1, res, subs, part_2)
    if modes[i] & FULL:
        subs[i] = words[i]
        dfs(words, modes, i + 1, res, subs, part_2)


def generate(words, modes, part_2):
    res = []
    subs = [u"" for _ in range(len(words))]
    dfs(words, modes, 0, res, subs, part_2)
    return res


def get_sep_by_lang(lang):
    if lang == 'zh':
        return u''
    else:
        return u' '


def strip_spans(s):
    return [t[0] for t in s]


def span_tokenize(s, lang = 'zh', character_per_span = False):
    if s == '':
        return []
    if character_per_span:
        tokens_with_spans = [(ch, i, i + 1) for i, ch in enumerate(s)]
    elif lang == 'zh':
        tokens_with_spans = merge_tokens(list(jieba.tokenize(s, mode = 'search')), s, keep_span = True)
    else:
        spans = WordPunctTokenizer().span_tokenize(s)
        tokens_with_spans = [(s[span[0]: span[1]], span[0], span[1]) for span in spans]
    return tokens_with_spans


def process(ent, mode_func, part_2):
    sub_ents = ent.split(u'/')
    res = []
    for sub_ent in sub_ents:
        words = merge_tokens(list(jieba.tokenize(sub_ent, mode = 'search')), sub_ent)
        modes = mode_func(words)
        aliases = generate(words, modes, part_2)
        res.extend(aliases)
    return list(set(res))


def load_from_short_file(name, full = False, th = -1, min_len = -1, use_mention2values = False):
    filename = '{}/{}.txt'.format(LEXICON_DIRECTORY, name)
    return load_from_file(filename, full, th, min_len, use_mention2values)


def load_from_file(filename, full = False, th = -1, min_len = -1, use_mention2values = False):
    if not use_mention2values:
        entities = []
    else:
        mention2values = dd(set)
    fin = utils.open_file(filename)
    line_cnt = 0
    print('Loading ', filename)
    for line in fin:
        if '\t' not in line:
            tks = line.strip().split(u':')
            if len(tks[0]) >= min_len:
                if not use_mention2values:
                    entities.append(tks[0])
                else:
                    mention2values[tks[0].lower()].add(tks[0].lower())
            if full and len(tks) > 1:
                for ent in tks[1].split(u','):
                    if len(ent) >= min_len:
                        if not use_mention2values:
                            entities.append(ent)
                        else:
                            mention2values[ent.lower()].add(tks[0].lower())
        else:
            tks = line.strip().split(u'\t')
            if len(tks) > 1:
                if tks[1].isdigit():
                    freq = int(tks[1])
                    if freq >= th and len(tks[0]) >= min_len:
                        if not use_mention2values:
                            entities.append(tks[0])
                        else:
                            mention2values[tks[0].lower()].add(tks[0].lower())
                    if freq < th:
                        break
            else:
                if len(tks[0]) >= min_len:
                    if not use_mention2values:
                        entities.append(tks[0])
                    else:
                        mention2values[tks[0].lower()].add(tks[0].lower())
        line_cnt += 1

        # if line_cnt == 100000: break # TODO: ONLY FOR DEBUG, SHOULD BE DELETED LATER
        # if line_cnt % 100000 == 0:
        #    print 'Loading line ', line_cnt
    fin.close()
    if not use_mention2values:
        print('lexicon size of', filename, ':', len(entities))
        return entities
    else:
        print('lexicon size of', filename, ':', len(mention2values))
        return mention2values


def load_mapping():
    mapping_filename = "{}/filename_entity.mapping".format(LEXICON_DIRECTORY)
    filename2entity = {}
    filename2entity_type = {}
    fin = utils.open_file(mapping_filename)
    for line in fin:
        if '#' in line:
            continue
        tks = line.strip().split('\t')
        if len(tks) < 2:
            print(line)
        filename2entity[tks[0]] = tks[1]
        if len(tks) == 3:
            filename2entity_type[tks[0]] = tks[2]
        else:
            filename2entity_type[tks[0]] = 'lexicon'
    fin.close()
    entity2filename = {v: k for k, v in filename2entity.items()}
    return filename2entity, entity2filename, filename2entity_type


class lexicon_generator:

    def __init__(self):
        self.freq_chat = utils.get_freq_chat()
        self.freq_general = utils.get_freq_general()

    def run(self, filename, mode_func, th_chat, th_general, th_lex, part_2 = True, debug = True, debug_file = 'log.txt', filter_set = [], complex_filter_set = [], exceptions = []):
        """
        filename (str): the path to the raw lexicon file
        mode_func (a function handle): the function that returns a list of modes given a list of words.
        th_chat (int): threshold for frequency in the chat corpus.
        th_general (int): threshold for frequency in the general corpus.
        th_lex (int): threshold for frequency in the current lexicon.
        part_2 (bool): whether the second character should be considered as the partial abbv. of a word.
        debug (bool): true to write to debug_file, false to write to filename.
        filter_set (list): a list of file paths indicating lexicons for filter.
        complex_filter_set (list): a list of file paths. Different from filter_set, complex_filter_set is used for AABB matching.
        exceptions (list): a list of word that will not be filtered by thresholds.
        """
        self.entities = load_from_file(filename)
        freq_lex = comp_term_freq(self.entities)

        alias2ent = dd(list)
        for i, ent in enumerate(self.entities):
            for alias in process(ent, mode_func, part_2):
                alias2ent[alias].append(ent)

        filter(alias2ent, self.freq_chat, th_chat, self.freq_general, th_general, freq_lex, th_lex, filter_set, complex_filter_set, exceptions)

        ent2alias = dd(list)
        for k, v in alias2ent.items():
            for vv in v:
                ent2alias[vv].append(k)
        for k in ent2alias.keys():
            ent2alias[k].sort(key = lambda x: len(x))

        filename = filename if not debug else debug_file
        fout = utils.open_file(filename, 'w')
        for ent, aliases in ent2alias.items():
            if len(aliases) > 0:
                fout.write(u"{}:{}\n".format(ent, u",".join(aliases)))
            else:
                fout.write(u"{}\n".format(ent))


def test():
    print(strip_spans(span_tokenize(u'河北考生理科男342')))

if __name__ == '__main__':
    test()


