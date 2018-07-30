
import numpy as np
from collections import defaultdict as dd

def parse_label(label):
    if label == 'O':
        return 'O', 'O'
    temp = label.split('-')
    return temp[0], temp[1]

def extract_entities(y, index2label, mask):
    entity_set = []
    for i in range(y.shape[0]):
        j = 0
        while j < y.shape[1] and mask[i, j] > 0:
            cur_parse = parse_label(index2label[y[i, j]])
            if cur_parse[0] in ['S', 'B']:
                k = j + 1
                while k < y.shape[1] and mask[i, k] > 0:
                    suc_parse = parse_label(index2label[y[i, k]])
                    if suc_parse[0] in ['I', 'E'] and suc_parse[1] == cur_parse[1]:
                        k += 1
                    else:
                        break
                entity_set.append((i, j, k, cur_parse[1]))
                j = k
            else:
                j += 1
    return set(entity_set)

def tagging_eval(y, py, mask, index2label):
    y_set = extract_entities(y, index2label, mask)
    py_set = extract_entities(py, index2label, mask)
    tp, fp = 0, 0
    for entity in py_set:
        if entity in y_set:
            tp += 1
        else:
            fp += 1
    fn = len(y_set) - tp
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    return f1

def text_eval(y, py):
    return 1.0 * np.array(y == py, dtype = np.int32).sum() / y.shape[0]

def text_eval_f1(y, py):
    tp, fp, fn = dd(int), dd(int), dd(int)
    class_cnt = y.max() + 1

    for i in range(y.shape[0]):
        if y[i] == py[i]:
            tp[y[i]] += 1
        else:
            fn[y[i]] += 1
            fp[py[i]] += 1

    f1 = 0
    for i in range(class_cnt):
        prec = 1.0 * tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] > 0 else 0.0
        recall = 1.0 * tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] > 0 else 0.0
        f1 += 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    return f1 / class_cnt



