# -*- coding: utf-8 -*-
from data.anno import *
from collections import defaultdict as dd
import lexicon_generator as lg
from config import *
import features.seg_feat as seg_feat
import features.tag_feat as tag_feat
import features.text_feat as text_feat
import utils

class Lexicon(Document):
    name = StringField()
    type = StringField()

class LexiconMention(Document):
    name = StringField()
    type = StringField()
    lexicon_id = ObjectIdField()

class sentence_struct:
    """data structure for a sentence. contains data structures used for feature extraction, including
    segment_struct, tagging_struct, and text_struct.

    tagging_inst: tagging_struct.
    segment_insts: list of segment_struct.
    """

    def __init__(self, q, lang = 'zh', tagging_inst = None, text_inst = None):
        """q: anno.Question,
        lang: str. zh or en.

        update tagging_inst, segment_insts, and text_inst.
        """

        if q is not None:
            n = len(q.text)
            tagging_label = ['O' for _ in range(n)]
            segment_insts = []

            intent = Intent.objects(id = q.intent_id).first()
            for e_mention in q.entity_mentions:
                entity_name = get_entity(e_mention, intent).name
                tagging_label[e_mention.start] = u"B-{}".format(entity_name)
                for i in range(e_mention.start + 1, e_mention.end):
                    tagging_label[i] = u"I-{}".format(entity_name)
                segment_insts.append(seg_feat.segment_struct(q.text, e_mention.start, e_mention.end, e_mention.role, entity_name, lang = lang))

            tagging_inst = tag_feat.tagging_struct(q.text, tagging_label, lang = lang)
            self.tagging_inst = tagging_inst
            self.segment_insts = segment_insts
            self.text_inst = text_feat.text_struct(q.text, intent.name, lang = lang)
        else:
            self.segment_insts = []
            self.tagging_inst = tagging_inst
            self.text_inst = text_inst

class data_manager:

    def __init__(self, business_id, lang = 'zh', intents = []):
        self.business_id = business_id
        if self.business_id is not None:
            self.load_data(lang = lang, intents = intents)

    def load_data(self, lang = 'zh', intents = []):
        self.sentence_insts = []

        for q in Question.objects(business_id = self.business_id):
            if q.annotated and q.has_intent:
                self.sentence_insts.append(sentence_struct(q, lang = lang))

        for intent in intents:
            self.sentence_insts.extend(intent.generate())

    def create_ner_dataset(self):
        tagging_insts = []
        for sentence_inst in self.sentence_insts:
            if sentence_inst.tagging_inst is not None:
                tagging_insts.append(sentence_inst.tagging_inst)
        return tagging_insts

    def create_role_labeling_dataset(self):
        ent2role = dd(set)

        for sentence_inst in self.sentence_insts:
            for segment_inst in sentence_inst.segment_insts:
                entity_name, role = segment_inst.entity_name, segment_inst.label
                ent2role[entity_name].add(role)

        ent_2_segment_insts = dd(list)
        for sentence_inst in self.sentence_insts:
            for segment_inst in sentence_inst.segment_insts:
                entity_name = segment_inst.entity_name
                if len(ent2role[entity_name]) < 2: continue
                ent_2_segment_insts[entity_name].append(segment_inst)
        return ent_2_segment_insts

    def create_intent_dataset(self):
        text_insts = []
        for sentence_inst in self.sentence_insts:
            text_insts.append(sentence_inst.text_inst)
        return text_insts

    def read_lexicon_set(self, type = '', keep_mention = True, from_file = False, filename = '', th = -1, min_len = -1):
        """name: str, name of the lexicon type,
        type: name of the lexicon type,
        keep_mention: bool, whether to read lexicon mentions,
        from_file: bool, whether to read the lexicon directly from file, used when reading large lexicon,
        filename: str, required when from_file is set to True,
        th: int, only read lexicons with frequency more than th. Will check the 2nd column of the tab seperated file,
        min_len: int, only read lexicons with length more than min_len.
        """
        ret = set()
        if not from_file:
            if keep_mention:
                for l_mention in LexiconMention.objects(type = type):
                    if len(l_mention.name) >= min_len:
                        ret.add(l_mention.name)
            else:
                for l in Lexicon.objects(type = type):
                    if len(l.name) >= min_len:
                        ret.add(l.name)
        else:
            if filename == '':
                utils.raise_error('Filename should not be empty when from_file is set to True!')
            ret = set(lg.load_from_short_file(filename, full = keep_mention, th = th, min_len = min_len))
        return ret

    def write_lexicon(self, data, type):
        """data: dict, key = name of lexicons, value = list of name of lexicon mentions; e.g., {u'清华大学': u'清华大学', u'清华'}
        type: name of the lexicon type
        """
        for k, v in data.items():
            lexicon = Lexicon.objects(name = k, type = type).first()
            if lexicon is None:
                lexicon = Lexicon(name = k, type = type)
                lexicon.save()

            # add the key, and dedup
            v.append(k)
            v = list(set(v))

            for name in v:
                l_mention = LexiconMention(name = name, type = type, lexicon_id = lexicon.id)
                l_mention.save()


def test():
    # b = Business.objects(name = 'edu.test').first()
    # dm = data_manager(b.id)
    # dm.load_data()
    # print 'len of sentence_insts', len(dm.sentence_insts)
    # print 'test done.'
    dm = data_manager(None)
    keywords = dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'aminer_keywords')
    print('len of lexicon', len(keywords))
    print('first few words:')
    cnt = 0
    for keyword in keywords:
        print(keyword)
        cnt += 1
        if cnt >= 10:
            break


if __name__ == '__main__':
    test()
