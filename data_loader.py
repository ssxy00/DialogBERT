# DialogBERT
# Copyright 2021-present NAVER Corp.
# BSD 3-clause

import os
import random
from copy import deepcopy
import numpy as np
import tables
import json
import itertools
from tqdm import tqdm
import torch
import torch.utils.data as data
import logging

logger = logging.getLogger(__name__)


class DialogTransformerDataset(data.Dataset):
    """
    A base class for Transformer dataset
    """

    def __init__(self, file_path, tokenizer,
                 min_num_utts=1, max_num_utts=7, max_utt_len=30,
                 block_size=256, utt_masklm=False, utt_sop=False,
                 context_shuf=False, context_masklm=False, do_test=False):
        self.do_test = do_test
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.min_num_utts = min_num_utts  # if not context_shuf and not context_masklm else 3
        self.max_num_utts = max_num_utts
        self.max_utt_len = max_utt_len
        self.block_size = block_size  # segment size to train BERT. when set -1 by default, use indivicual sentences(responses) as BERT inputs.
        # Otherwise, clip a block from the context.

        self.utt_masklm = utt_masklm
        self.utt_sop = utt_sop
        self.context_shuf = context_shuf
        self.context_masklm = context_masklm

        self.rand_utt = [tokenizer.mask_id] * (max_utt_len - 1) + [tokenizer.eos_id]  # update during loading

        # a cache to store context and response that are longer than min_num_utts
        self.cache = [[tokenizer.mask_id] * max_utt_len] * max_num_utts, [tokenizer.mask_id] * max_utt_len

        self.perm_list = [list(itertools.permutations(range(L))) for L in range(1, max_num_utts + 1)]
        print("loading data...")
        table = tables.open_file(file_path)
        self.contexts = table.get_node('/sentences')[:].astype(np.long)
        # self.knowlege = table.get_node('/knowledge')[:].astype(np.long)
        self.index = table.get_node('/indices')[:]
        self.data_len = self.index.shape[0]
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        index = self.index[offset]
        pos_utt, ctx_len, res_len, = index['pos_utt'], index['ctx_len'], index['res_len']
        # pos_knowl, knowl_len = index['pos_knowl'], index['knowl_len']

        ctx_len = min(ctx_len, self.block_size) if self.block_size > -1 else ctx_len  # trunck too long context

        ctx_arr = self.contexts[pos_utt - ctx_len:pos_utt].tolist()
        res_arr = self.contexts[pos_utt:pos_utt + res_len].tolist()
        # knowl_arr = self.knowledge[pos_knowl:pos_knowl+knowl_len].tolist()

        ## split context array into utterances        
        context = []
        tmp_utt = []
        for i, tok in enumerate(ctx_arr):
            tmp_utt.append(ctx_arr[i])
            if tok == self.tokenizer.eos_id:
                floor = tmp_utt[0]
                tmp_utt = tmp_utt[1:-1]  # remove floor and eos
                utt_len = min(len(tmp_utt), self.max_utt_len)
                utt = [self.tokenizer.bos_id] + tmp_utt[:utt_len] + [self.tokenizer.eos_id]
                context.append(utt)  # append utt to context          
                tmp_utt = []  # reset tmp utt
        response = res_arr[1:-1]  # remove floor and eos
        res_len = len(response)
        if not self.do_test:
            res_len = min(len(response), self.max_utt_len)
        response = [self.tokenizer.bos_id] + response[:res_len] + [self.tokenizer.eos_id]

        num_utts = min(len(context), self.max_num_utts)
        context = context[-num_utts:]

        return context, response  # , knowledge

    def list2array(self, L, d1_len, d2_len=0, d3_len=0, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''

        def list_dim(a):
            if type(a) != list:
                return 0
            elif len(a) == 0:
                return 1
            else:
                return list_dim(a[0]) + 1

        if type(L) is not list:
            print("requires a (nested) list as input")
            return None

        if list_dim(L) == 0:
            return L
        elif list_dim(L) == 1:
            arr = np.zeros(d1_len, dtype=dtype) + pad_idx
            for i, v in enumerate(L): arr[i] = v  # ssxy: L 的长度不可能大于 d1_len，否则会报错
            return arr
        elif list_dim(L) == 2:
            arr = np.zeros((d2_len, d1_len), dtype=dtype) + pad_idx
            for i, row in enumerate(L):
                for j, v in enumerate(row):
                    arr[i][j] = v
            return arr
        elif list_dim(L) == 3:
            arr = np.zeros((d3_len, d2_len, d1_len), dtype=dtype) + pad_idx
            for k, group in enumerate(L):
                for i, row in enumerate(group):
                    for j, v in enumerate(row):
                        arr[k][i][j] = v
            return arr
        else:
            print('error: the list to be converted cannot have a dimenson exceeding 3')

    def mask_context(self, context):
        def is_special_utt(utt):
            return len(utt) == 3 and utt[1] in [self.tokenizer.mask_id, self.tokenizer.eos_id, self.tokenizer.bos_id]

        utts = [utt for utt in context]
        lm_label = [[-100] * len(utt) for utt in context]
        context_len = len(context)
        assert context_len > 1, 'a context to be masked should have at least 2 utterances'

        mlm_probs = [0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
        mlm_prob = mlm_probs[context_len - 1]

        prob = random.random()
        if prob < mlm_prob:
            i = random.randrange(context_len)
            while is_special_utt(utts[i]):
                i = random.randrange(context_len)
            utt = utts[i]
            prob = prob / mlm_prob
            if prob < 0.8:  # 80% randomly change utt to mask utt
                utts[i] = [self.tokenizer.bos_id, self.tokenizer.mask_id, self.tokenizer.eos_id]
            elif prob < 0.9:  # 10% randomly change utt to a random utt
                utts[i] = deepcopy(self.rand_utt)
            lm_label[i] = deepcopy(utt)
            # assert len(utts[i]) == len(lm_label[i]), "the size of the lm label is different to that of the masked utterance"
            self.rand_utt = deepcopy(utt)  # update random utt  # ssxy: interesting implementation
        return utts, lm_label

    def shuf_ctx(self, context):
        perm_label = 0  # ssxy: the idx of permutation in permutation list ([0], [0, 1], [1, 0], ...)
        num_utts = len(context)
        if num_utts == 1:
            return context, perm_label, [0]
        for i in range(num_utts - 1): perm_label += len(self.perm_list[i])
        perm_id = int(random.random() * len(self.perm_list[num_utts - 1]))
        perm_label += perm_id
        ctx_position_ids = self.perm_list[num_utts - 1][perm_id]
        # new context
        shuf_context = [context[i] for i in ctx_position_ids]
        return shuf_context, perm_label, ctx_position_ids

    def __len__(self):
        return self.data_len


class HBertMseEuopDataset(DialogTransformerDataset):
    """
    A hierarchical Bert data loader where the context is masked with ground truth utterances and to be trained with MSE matching.
    The context is shuffled for a novel energy-based order prediction approach (EUOP)
    """

    def __init__(self, file_path, tokenizer,
                 min_num_utts=1, max_num_utts=9, max_utt_len=30,  # ssxy: why 9? including cls_utt and sep_utt
                 block_size=-1, utt_masklm=False, utt_sop=False,
                 context_shuf=False, context_masklm=False, do_test=False):

        super(HBertMseEuopDataset, self).__init__(
            file_path, tokenizer, min_num_utts, max_num_utts, max_utt_len, block_size, utt_masklm, utt_sop,
            context_shuf, context_masklm, do_test)

        self.cls_utt = [tokenizer.bos_id, tokenizer.bos_id, tokenizer.eos_id]
        self.sep_utt = [tokenizer.bos_id, tokenizer.eos_id, tokenizer.eos_id]
        self.do_test = do_test

    def __getitem__(self, offset):
        context, response = super().__getitem__(offset)

        context_len = min(len(context), self.max_num_utts - 2)
        context = [self.cls_utt] + context[-context_len:] + [self.sep_utt]
        context_len += 2
        context_attn_mask = [1] * context_len
        context_mlm_target = [[-100] * len(utt) for utt in context]
        context_position_perm_id = -100
        context_position_ids = list(range(context_len))  #

        if self.context_shuf and random.random() < 0.4 and len(context) > 2:
            context_, context_position_perm_id, context_position_ids_ = self.shuf_ctx(context[1:-1])
            context = [self.cls_utt] + context_ + [self.sep_utt]
            context_position_ids = [0] + [p + 1 for p in context_position_ids_] + [context_len - 1]
            context_mlm_target = [[-100] * len(utt) for utt in context]

        if self.context_masklm and context_position_perm_id < 2 and len(
                context) > 4:  # ssxy: context_position_perm_id < 2 means no permutation is applied?
            context, context_mlm_target = self.mask_context(context)

        context_utts_attn_mask = [[1] * len(utt) for utt in context]

        # ssxy: padding
        context = self.list2array(context, self.max_utt_len + 2, self.max_num_utts, pad_idx=self.tokenizer.pad_id)
        context_utts_attn_mask = self.list2array(context_utts_attn_mask, self.max_utt_len + 2, self.max_num_utts)
        context_attn_mask = self.list2array(context_attn_mask, self.max_num_utts)
        context_mlm_target = self.list2array(context_mlm_target, self.max_utt_len + 2, self.max_num_utts, pad_idx=-100)
        context_position_ids = self.list2array(context_position_ids, self.max_num_utts)

        if self.do_test:
            response = self.list2array(response, len(response), pad_idx=self.tokenizer.pad_id)  # for decoder training
        else:
            response = self.list2array(response, self.max_utt_len + 2, pad_idx=self.tokenizer.pad_id)  # for decoder training

        return context, context_utts_attn_mask, context_attn_mask, \
               context_mlm_target, context_position_perm_id, context_position_ids, response


def load_dict(filename):
    return json.loads(open(filename, "r").readline())


def load_vecs(fin):
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs = h5f.root.vecs

    vecs = np.zeros(shape=h5vecs.shape, dtype=h5vecs.dtype)
    vecs[:] = h5vecs[:]
    h5f.close()
    return vecs


def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root, 'vecs', atom, vecs.shape, filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()
