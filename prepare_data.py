# DialogBERT
# Copyright 2021-present NAVER Corp.
# BSD 3-clause

import random
import numpy as np
import argparse
import json
import tables
import os
import re
from tqdm import tqdm
import pickle as pkl
from vocabs.gpt2_tokenizer import GPT2Vocab

from data_loader import load_dict, save_vecs

class Index(tables.IsDescription):
    pos_utt = tables.Int32Col()  # start offset of an utterance
    res_len = tables.Int32Col()  # number of tokens till the end of response
    ctx_len = tables.Int32Col()  # number of tokens from the start of dialog


def binarize(dialogs, tokenizer, output_path):
    """binarize data and save the processed data into a hdf5 file
       :param dialogs: an array of dialogs, 
        each element is a list of <caller, utt, feature> where caller is a string of "A" or "B",
        utt is a sentence, feature is an 2D numpy array 
    """
    # ssxy: equal to our split method, but this saving method is more efficient because
    # they save sentences and indices separately
    # {(x_1), x_2}, {(x_1, x_2), x_3}, ...
    f = tables.open_file(output_path, 'w')
    filters = tables.Filters(complib='blosc', complevel=5)
    arr_contexts = f.create_earray(f.root, 'sentences', tables.Int32Atom(), shape=(0,), filters=filters)
    indices = f.create_table("/", 'indices', Index, "a table of indices and lengths")
    pos_utt = 0
    for i, dialog in enumerate(tqdm(dialogs)):

        n_tokens = 0
        ctx_len = 0
        for k, (caller, utt, feature) in enumerate(dialog['utts']):
            floor = -1 if caller == 'A' else -2
            idx_utt = tokenizer.string2ids(utt) + [tokenizer.eos_id]  # ssxy: 这里的 eos_id 是用来在数据预处理时断句的，在数据预处理时还会再次处理 bos 和 eos
            arr_contexts.append([floor])
            arr_contexts.append(idx_utt)
            n_tokens += len(idx_utt) + 1
            if k >= 7:  # ignore the first seven utterances
                ind = indices.row
                ind['pos_utt'] = pos_utt
                ind['res_len'] = len(idx_utt) + 1
                ind['ctx_len'] = ctx_len
                ind.append()
            ctx_len += len(idx_utt) + 1
            pos_utt += len(idx_utt) + 1
    f.close()


def get_daily_dial_data(data_path):
    dialogs = []
    dials = open(data_path, 'r').readlines()
    for dial in dials:
        dial = dial.strip()
        if len(dial) == 0:
            continue
        dial = [seq.strip() for seq in dial.split('__eou__')[:-1]]
        if len(dial) <= 7:
            continue
        utts = []
        for i, utt in enumerate(dial):
            caller = 'A' if i % 2 == 0 else 'B'
            utts.append((caller, utt, np.zeros((1, 1))))
        dialog = {'knowledge': '', 'utts': utts}
        dialogs.append(dialog)
    return dialogs


def get_multiwoz_data(data_path):
    timepat = re.compile("\d{1,2}[:]\d{1,2}")
    pricepat = re.compile("\d{1,3}[.]\d{1,2}")

    def normalize(text):
        text = text.lower()
        text = re.sub(r'^\s*|\s*$', '', text)  # replace white spaces in front and end
        # hotel domain pfb30
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(': sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))
        # normalize postcode
        ms = re.findall(
            '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]
        text = re.sub(u"(\u2018|\u2019)", "'", text)  # weird unicode bug
        # replace time and and price
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [value_price] ', text)
        # text = re.sub(pricepat2, '[value_price]', text)
        # replace st.
        text = text.replace(';', ',')
        text = re.sub('$\/', '', text)
        text = text.replace('/', ' and ')
        # replace other special characters
        text = text.replace('-', ' ')
        text = re.sub('[\":\<>@\(\)]', '', text)
        text = re.sub(' +', ' ', text)  # remove multiple spaces
        # concatenate numbers
        tmp = text
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(u'^\d+$', tokens[i]) and re.match(u'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)
        return text

    dialogs = []

    data = json.load(open(data_path, 'r'))
    for dialogue_name in tqdm(data):
        utts = []
        dialogue = data[dialogue_name]
        caller = 'A'
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])
            utts.append((caller, sent, np.zeros((1, 1))))
            caller = 'B' if caller == 'A' else 'A'
        dialog = {'knowledge': '', 'utts': utts}
        dialogs.append(dialog)

    return dialogs[:-2000], dialogs[-2000:-1000], dialogs[-1000:]


def get_personachat_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        dialogs = []
        utts = []
        for line in tqdm(file.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])
            if int(dialog_idx) == 1:
                if len(utts):
                    dialog = {'knowledge': '', 'utts': utts}
                    dialogs.append(dialog)
                utts = []
                caller = 'A'

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
            if dialog_line[0].startswith('partner\'s persona:'):
                persona_info = dialog_line[0].replace('partner\'s persona: ', '')
            elif len(dialog_line) > 1:
                utts.append((caller, dialog_line[0], np.zeros((1, 1))))
                caller = 'B' if caller == 'A' else 'A'
                utts.append((caller, dialog_line[1], np.zeros((1, 1))))
                caller = 'B' if caller == 'A' else 'A'
        if len(utts):
            dialog = {'knowledge': '', 'utts': utts}
            dialogs.append(dialog)
    return dialogs

def load_data(data_name):
    data = {'train': [], 'valid': [], 'test': []}
    if data_name == 'dailydialog':
        data_dir = "/home1/sxy/datasets/DailyDialog/ijcnlp_dailydialog/"

        data['train'] = get_daily_dial_data(data_dir + 'train/dialogues_train.txt')
        data['valid'] = get_daily_dial_data(data_dir + 'validation/dialogues_validation.txt')
        data['test'] = get_daily_dial_data(data_dir + 'test/dialogues_test.txt')

    elif data_name == 'multiwoz':
        data_dir = "/home1/sxy/datasets/MultiWOZ/MultiWOZ_1.0"
        train, valid, test = get_multiwoz_data(os.path.join(data_dir, 'data.json'))
        data['train'] = train
        data['valid'] = valid
        data['test'] = test

    elif data_name == "personachat":
        data["train"] = get_personachat_data("/home1/sxy/datasets/ConvAI2_data/ConvAI2/train_both_original_no_cands.txt")
        valid_and_test_data = get_personachat_data("/home1/sxy/datasets/ConvAI2_data/ConvAI2/valid_both_original_no_cands.txt")
        random.seed(0)
        random.shuffle(valid_and_test_data)
        n_valid_data = len(valid_and_test_data)
        data["valid"] = valid_and_test_data[: n_valid_data // 2]
        data["test"] = valid_and_test_data[n_valid_data // 2:]

    return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_set", default='personachat', help='multiwoz, dailydialog, personachat')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print("loading data...")
    data = load_data(args.data_set)

    train_data = data["train"]
    valid_data = data["valid"]
    test_data = data["test"]

    tokenizer = GPT2Vocab(model_path="/home1/sxy/models/transformers3_gpt2-small")

    output_data_dir = f"/home1/sxy/DialogBERT/datasets/{args.data_set}"

    print('binarizing training data')
    train_out_path = os.path.join(output_data_dir, "train.h5")
    train_data_binary = binarize(train_data, tokenizer, train_out_path)

    print('binarizing validation data')
    dev_out_path = os.path.join(output_data_dir, "valid.h5")
    dev_data_binary = binarize(valid_data, tokenizer, dev_out_path)

    print('binarizing test data')
    test_out_path = os.path.join(output_data_dir, "test.h5")
    test_data_binary = binarize(test_data, tokenizer, test_out_path)
