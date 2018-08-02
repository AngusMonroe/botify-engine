# -*- coding: utf-8 -*-

import save_db_from_json
from simulator import *
import lexicon_generator as lg
from data_manager import *


def simulate(debug=False, debug_file='log.txt'):
    """
    debug (bool): True to print sentences to file, False to save it to db
    debug_file (str): the file name to save the debug output
    """

    dm = data_manager(None)
    keywords = list(
        dm.read_lexicon_set(keep_mention=False, from_file=True, filename='aminer_keywords', th=1000, min_len=3))
    insts = list(dm.read_lexicon_set(keep_mention=False, from_file=True, filename='linkedin_inst', th=1000, min_len=3))
    names = list(dm.read_lexicon_set(keep_mention=False, from_file=True, filename='aminer_names', th=10, min_len=5))
    years = list(dm.read_lexicon_set(keep_mention=False, from_file=True, filename='year'))
    venues = list(dm.read_lexicon_set(keep_mention=False, from_file=True, filename='bigsci_venues', th=1000, min_len=3))

    # keywords = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'aminer_keywords'))
    # insts = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'linkedin_inst'))
    # names = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'aminer_names'))
    # years = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'year'))
    # venues = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'dblp_venues'))

    f2e, e2f, _ = lg.load_mapping()
    keyword_node = node(keywords, entity=f2e['aminer_keywords'])
    inst_node = node(insts, entity=f2e['linkedin_inst'])
    name_node = node(names, entity=f2e['aminer_names'])
    year_node = node(years, entity=f2e['year'])
    venue_node = node(venues, entity=f2e['dblp_venues'])

    # begin intent search expert #
    front_keyword_node = node(p_dropout=0.5).add_child(keyword_node)
    front_inst_node = node(p_dropout=0.5).add_child(inst_node)
    expert_node = node([u"researchers", u"scientists", u"people", u"professors", u"experts"], p_dropout=0.0, lang='en',
                       gen_lemmas=True)

    front_cond_node = node(exchangeable=True, lang='en').add_child(front_keyword_node).add_child(front_inst_node)

    which_node = node([u"that", u"who"])
    rear_keyword_node_1 = node(lang='en').add_child(which_node).add_child(node(
        [u"work on", u"are working on", u"are doing", u"do", u"are doing research on", u"are experts at",
         u"have conducted research on", u"have been working on"], lang='en', gen_lemmas=True)).add_child(keyword_node)
    rear_keyword_node_2 = node(lang='en').add_child(
        node([u"working on", u"doing", u"doing research on", u"conducting research on"])).add_child(keyword_node)
    rear_keyword_node_3 = node(lang='en').add_child(node(
        [u"whose work", u"whose research", u"whose paper", u"whose works", u"whose papers",
         u"whose researches"])).add_child(
        node([u"focus on", u"are about", u"are in", u"are related to"], lang='en', gen_lemmas=True)).add_child(
        keyword_node)
    rear_keyword_node = node(pick_one=True, p_dropout=0.5).add_child(rear_keyword_node_1, 0.2).add_child(
        rear_keyword_node_2, 0.6).add_child(rear_keyword_node_3, 0.2)
    rear_inst_node_1 = node(lang='en').add_child(which_node).add_child(
        node([u"are from", u"work at", u"are in", u"work in", u"are at"], lang='en', gen_lemmas=True)).add_child(
        inst_node)
    rear_inst_node_2 = node(lang='en').add_child(node([u"from", u"working at", u"in", u"at"])).add_child(inst_node)
    rear_inst_node = node(pick_one=True, p_dropout=0.5).add_child(rear_inst_node_1, 0.2).add_child(rear_inst_node_2,
                                                                                                   0.6)

    rear_cond_node = node(exchangeable=True, lang='en').add_child(rear_inst_node).add_child(rear_keyword_node)
    conded_expert_node = node(lang='en').add_child(front_cond_node).add_child(expert_node).add_child(rear_cond_node)

    search_node_1 = node([u"are there any", u"are there", u"who are", u"what are"], lang='en', gen_lemmas=True)
    search_node_2 = node(lang='en').add_child(node(
        [u"give me", u"want to find", u"wanna find", u"find", u"find for", u"search", u"search for", u"query", u"show",
         u"look up for"])).add_child(
        node([u"the", u"those", u"the group of", u"a group of", u"some", u"a number of", u"a list of", u"the list of"],
             p_dropout=0.8))
    search_node = node(pick_one=True, p_dropout=0.5).add_child(search_node_1).add_child(search_node_2)

    search_expert_node = node(lang='en').add_child(search_node).add_child(conded_expert_node)
    # end intent search expert #

    # begin intent search paper #
    front_keyword_node = node(p_dropout=0.5).add_child(keyword_node)
    front_inst_node = node(p_dropout=0.5).add_child(inst_node)
    front_year_node = node(p_dropout=0.5).add_child(year_node)
    front_name_node = node(p_dropout=0.5).add_child(name_node).add_child(node(u"'s", p_dropout=0.5))
    front_venue_node = node(p_dropout=0.5).add_child(venue_node)
    paper_node = node([u"papers", u"works", u"journals", u"publications", u"researches"], p_dropout=0.0, lang='en',
                      gen_lemmas=True)

    front_cond_node = node(exchangeable=True, lang='en').add_child(front_keyword_node).add_child(
        front_inst_node).add_child(front_year_node).add_child(front_name_node).add_child(front_venue_node)

    which_node = node([u"that", u"which"])
    rear_keyword_node_1 = node(lang='en').add_child(which_node).add_child(
        node([u"focus on", u"are about", u"are related to"], lang='en', gen_lemmas=True)).add_child(keyword_node)
    rear_keyword_node_2 = node(lang='en').add_child(node([u"focusing on", u"on", u"about", u"related to"])).add_child(
        keyword_node)
    rear_keyword_node = node(pick_one=True, p_dropout=0.5).add_child(rear_keyword_node_1, 0.2).add_child(
        rear_keyword_node_2, 0.6)
    rear_inst_node_1 = node(lang='en').add_child(which_node).add_child(
        node([u"are from", u"are by", u"are written by", u"are made by", u"are done by", u"are published by"],
             lang='en', gen_lemmas=True)).add_child(inst_node)
    rear_inst_node_2 = node(lang='en').add_child(
        node([u"from", u"by", u"written by", u"made by", u"done by", u"published by"])).add_child(inst_node)
    rear_inst_node = node(pick_one=True, p_dropout=0.5).add_child(rear_inst_node_1, 0.2).add_child(rear_inst_node_2,
                                                                                                   0.6)
    rear_name_node_1 = node(lang='en').add_child(which_node).add_child(
        node([u"are from", u"are by", u"are written by", u"are made by", u"are done by", u"are published by"],
             lang='en', gen_lemmas=True)).add_child(name_node)
    rear_name_node_2 = node(lang='en').add_child(
        node([u"from", u"by", u"written by", u"made by", u"done by", u"published by"])).add_child(name_node)
    rear_name_node = node(pick_one=True, p_dropout=0.5).add_child(rear_name_node_1, 0.2).add_child(rear_name_node_2,
                                                                                                   0.6)
    rear_year_node_1 = node(lang='en').add_child(which_node).add_child(node(
        [u"are in", u"are written in", u"are made in", u"are done in", u"are published in", u"are at",
         u"are written at", u"are made at", u"are done at", u"are published at"], lang='en',
        gen_lemmas=True)).add_child(year_node)
    rear_year_node_2 = node(lang='en').add_child(node(
        [u"in", u"written in", u"made in", u"done in", u"published in", u"at", u"written at", u"made at", u"done at",
         u"published at"])).add_child(year_node)
    rear_year_node = node(pick_one=True, p_dropout=0.5).add_child(rear_year_node_1, 0.2).add_child(rear_year_node_2,
                                                                                                   0.6)
    rear_venue_node_1 = node(lang='en').add_child(which_node).add_child(
        node([u"appear in", u"appear on", u"are on", u"are published on", u"are received by", u"are accepted by"],
             lang='en', gen_lemmas=True)).add_child(venue_node)
    rear_venue_node_2 = node(lang='en').add_child(
        node([u"on", u"appearing in", u"appearing on", u"published on", u"received by", u"accepted by"])).add_child(
        venue_node)
    rear_venue_node = node(pick_one=True, p_dropout=0.5).add_child(rear_venue_node_1, 0.2).add_child(rear_venue_node_2,
                                                                                                     0.6)

    rear_cond_node = node(exchangeable=True, lang='en').add_child(rear_keyword_node).add_child(
        rear_inst_node).add_child(rear_name_node).add_child(rear_year_node).add_child(rear_venue_node)
    conded_paper_node = node(lang='en').add_child(front_cond_node).add_child(paper_node).add_child(rear_cond_node)

    search_node_1 = node([u"are there any", u"are there", u"what are", u"which are"], lang='en', gen_lemmas=True)
    search_node_2 = node(lang='en').add_child(node(
        [u"give me", u"want to find", u"wanna find", u"find", u"find for", u"search", u"search for", u"query", u"show",
         u"look up for"])).add_child(
        node([u"the", u"those", u"the group of", u"a group of", u"some", u"a number of", u"a list of", u"the list of"],
             p_dropout=0.8))
    search_node = node(pick_one=True, p_dropout=0.5).add_child(search_node_1).add_child(search_node_2)

    search_paper_node = node(lang='en').add_child(search_node).add_child(conded_paper_node)
    # end intent search paper #

    # begin intent search venue #
    front_keyword_node = node(p_dropout=0.5).add_child(keyword_node)
    venue_node = node([u"journals", u"conferences"], p_dropout=0.0, lang='en', gen_lemmas=True)

    front_cond_node = node(exchangeable=True, lang='en').add_child(front_keyword_node)

    which_node = node([u"that", u"which"])
    rear_keyword_node_1 = node(lang='en').add_child(which_node).add_child(
        node([u"focus on", u"are about", u"are related to"], lang='en', gen_lemmas=True)).add_child(keyword_node)
    rear_keyword_node_2 = node(lang='en').add_child(node([u"focusing on", u"on", u"about", u"related to"])).add_child(
        keyword_node)
    rear_keyword_node_3 = node(lang='en').add_child(
        node([u"whose papers", u"of which the papers", u"on which the papers"], lang='en', gen_lemmas=True)).add_child(
        node([u"focus on", u"are about", u"are in", u"are related to"], lang='en', gen_lemmas=True)).add_child(
        keyword_node)
    rear_keyword_node = node(pick_one=True, p_dropout=0.5).add_child(rear_keyword_node_1, 0.2).add_child(
        rear_keyword_node_2, 0.6).add_child(rear_keyword_node_3, 0.2)

    rear_cond_node = node(exchangeable=True, lang='en').add_child(rear_keyword_node)
    conded_venue_node = node(lang='en').add_child(front_cond_node).add_child(venue_node).add_child(rear_cond_node)

    search_node_1 = node([u"are there any", u"are there", u"which are", u"what are"], lang='en', gen_lemmas=True)
    search_node_2 = node(lang='en').add_child(node(
        [u"give me", u"want to find", u"wanna find", u"find", u"find for", u"search", u"search for", u"query", u"show",
         u"look up for"])).add_child(
        node([u"the", u"those", u"the group of", u"a group of", u"some", u"a number of", u"a list of", u"the list of"],
             p_dropout=0.8))
    search_node = node(pick_one=True, p_dropout=0.5).add_child(search_node_1).add_child(search_node_2)

    search_venue_node = node(lang='en').add_child(search_node).add_child(conded_venue_node)
    # end intent search venue #


    s = simulator().add_root(search_expert_node, u"搜学者", 0.6).add_root(search_paper_node, u"搜文章", 0.6).add_root(
        search_venue_node, u"搜会议", 0.3)
    annos = s.generate(10000)
    if debug:
        # fout = utils.open_file(debug_file, 'w')
        f1 = utils.open_file('entity.txt', 'w')
        # f2 = utils.open_file('out.txt', 'w')
        for anno in annos:
            # fout.write(u"{} {}\n".format(anno['intent'], anno['question']['text']))
            for item in anno['entity_mentions']:
                if item:
                    f1.write(u"{} {}\n".format(item['snippet'], item['entity']))
            f1.write('\n')
            # f2.write(u"{}\n".format(anno['question']['text']))
    else:
        save_db_from_json.test('aminer@aminer.org', 'scholar.test', None, annos=annos)


def test():
    simulate(debug=True)


if __name__ == '__main__':
    test()
