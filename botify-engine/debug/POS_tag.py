# -*- coding: utf-8 -*-
f1 = open('../entity.txt', 'r', encoding='utf8')
# f2 = open('out.txt', 'r', encoding='utf8')
f_out = open('../data.txt', 'w', encoding='utf8')

flag = 0
word = ''
tag = ''
for line in f1.readlines():
    item = line.split()
    # entity = item[-1].split('-')

    if len(item) <= 1:
        flag = 0
        f_out.write(word + ' ' + tag + '\n')
        print(word)
        word = ''
        tag = ''
        if len(item) == 0:
            f_out.write('\n')
        continue

    if item[1] != tag and item[1] == 'O' and tag != '':
        flag = 0
        f_out.write(word + ' ' + tag + '\n')
        print(word)
        word = ''

    if flag == 0:
        flag = 1
        tag = item[1]

    word += item[0]

f1.close()
f_out.close()

print("done")
