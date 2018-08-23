# -*- coding: utf-8 -*-

import random

train_file = open("data/aminer/aminer_train.dat", "w", encoding="utf8")
dev_file = open("data/aminer/aminer_dev.dat", "w", encoding="utf8")
test_file = open("data/aminer/aminer_test.dat", "w", encoding="utf8")
train_test_file = open("data/aminer/aminer_train_test.dat", "w", encoding="utf8")


def add_tag(data_path, tag):
    print('dealing with ' + tag + '...')
    file = open(data_path, 'r', encoding='utf8')
    for line in file.readlines():
        num = random.randint(1, 100)
        file = train_file
        if num <= 4:
            file = train_test_file
        elif num <= 12:
            file = test_file
        elif num <= 20:
            file = dev_file

        word = line.split()
        for i in range(len(word) - 1):
            out_str = ''
            if i == 0:
                out_str += 'B-' + tag
            else:
                out_str += 'I-' + tag
            # print(out_str)
            file.write(str(word[i]) + ' ' + out_str + '\n')
        file.write('\n')
    print(tag + ' data has already done.')


def add_date(data_path, tag):
    print('dealing with ' + tag + ' tag...')
    file = open(data_path, 'r', encoding='utf8')
    for line in file.readlines():
        num = random.randint(1, 100)
        file = train_file
        if num <= 4:
            file = train_test_file
        elif num <= 12:
            file = test_file
        elif num <= 20:
            file = dev_file

        word = line.split(':')
        if word:
            out_str = str(word[0].strip('\n')) + ' ' 'B-' + tag
            file.write(out_str + '\n\n')
    print(tag + ' data has already done.')

if __name__ == '__main__':
    add_tag('data/lexicons/aminer_names.txt', 'PER')
    add_tag('data/lexicons/aminer_keywords.txt', 'KEY')
    add_tag('data/lexicons/bigsci_venues.txt', 'CON')
    add_tag('data/lexicons/linkedin_inst.txt', 'ORG')
    add_date('data/lexicons/year.txt', 'DATE')

    train_file.write('-DOCSTART- -X- O O')
    train_test_file.write('-DOCSTART- -X- O O')
    dev_file.write('-DOCSTART- -X- O O')
    test_file.write('-DOCSTART- -X- O O')

    train_test_file.close()
    train_file.close()
    test_file.close()
    dev_file.close()
