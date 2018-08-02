# -*- coding:utf-8 -*-

import pymongo
from urllib import parse

passwd = input("Please input the password:")
passwd = parse.quote(passwd)  # 对密码先进行编码
mango_uri = 'mongodb://%s:%s@%s:%s/%s' % ("kegger", passwd, "166.111.7.173", "30019", "admin")
client = pymongo.MongoClient(mango_uri, unicode_decode_error_handler='ignore')

# 操作test数据库
db_name = 'bigsci'
db = client[db_name]

# collist = db.collection_names()
# print(collist)

collection_set01 = db['publication_dupl']

dic = {}
for i, r in enumerate(collection_set01.find({'lang': 'en'})):
        if 'venue' in r.keys():
            if 'raw' in r['venue'].keys():
                if r['venue']['raw'] in dic.keys():
                    dic[r['venue']['raw']] += 1
                else:
                    dic[r['venue']['raw']] = 1
        print(str(r['_id']) + ' ' + str(i))

out_file = open("../data/lexicons/bigsci_venues.txt", "w", encoding="utf8")
for key in dic.keys():
    out_file.write(key + ' ' + str(dic[key]) + '\n')

print("done")
