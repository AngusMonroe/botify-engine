# -*- coding: utf-8 -*-
import json
from data.anno import *
# from db import mongo
import mongoengine
import datetime
import utils
from config import *

def test(b_email, b_name, filename, annos = None, no_data = False):

    # mongo.connect()

    b = Business(email=b_email, name=b_name)
    print(b.name)
    raise NotImplemented

    if b is not None:
        Intent.objects(business_id = b.id).delete()
        Entity.objects(business_id = b.id).delete()
        Question.objects(business_id = b.id).delete()
    else:
        b = Business(email = b_email, name = b_name)
        # b.save()

    if no_data:
        return

    if annos is None:
        # annos = json.load(utils.open_file('engine/data/jsons/{}.annotations.json'.format(b_name)))
        annos = json.load(utils.open_file(filename))
    for index, anno in enumerate(annos):
        if index % 1000 == 0:
            print('writing', index)
        e_mentions = []
        i = Intent.objects(name = anno['intent'], business_id = b.id).first()
        if i is None:
            i = Intent(name = anno['intent'], business_id = b.id)
        i_roles = i.roles.copy()
        for mention in anno['entity_mentions']:
            e_mentions.append(EntityMention(role = mention['role'], start = mention['start'], end = mention['end']))
            
            e = Entity.objects(name = mention['entity'], business_id = b.id).first()
            if e is None:
                e = Entity(name = mention['entity'], business_id = b.id)
                e.save()

            i_roles[mention['role']] = e.id
        i.roles = i_roles
        i.save()

        q = Question(text = anno['question']['text'],
            entity_mentions = e_mentions,
            has_intent = True,
            intent_id = i.id,
            upload_time = datetime.datetime.now(),
            annotated = True,
            business_id = b.id
        )
        q.save()

    print('test done.')

if __name__ == '__main__':
    # test('jh@jhhrm.com', 'edu.test', '{}/edu.test.annotations.json'.format(JSON_DIRECTORY))
    test('Marketing@gedu.org', 'ielts.test', '', no_data = True)
