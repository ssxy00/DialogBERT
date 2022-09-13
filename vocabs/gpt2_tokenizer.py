# -*- coding: utf-8 -*-
# @Time        : 2022/9/7 15:06
# @Author      : ssxy00
# @File        : gpt2_tokenizer.py
# @Description :

# 相比 GLCM 中的 vocab，这里新增了 MASK，为了保持词表大小一致，删除了目前用不到的 <fact>，这样应该是可以直接和之前结果对比。
# TODO 未来也可以再统一词表跑一下实验

from transformers import GPT2Tokenizer

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'mask_token': '<mask>',
                         'additional_special_tokens': ("<bot>", "<human>")}
SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', "<bot>", "<human>"]

class GPT2Vocab:

    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.add_special_tokens_()


    def add_special_tokens_(self):
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        orig_num_tokens = len(self.tokenizer.encoder)
        # print(orig_num_tokens)
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
        self.special_tokens = SPECIAL_TOKENS
        # print(num_added_tokens)

    def __len__(self):
        return len(self.tokenizer)

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    @property
    def mask_id(self):
        return self.tokenizer.mask_token_id

    @property
    def bot_id(self):
        return self.tokenizer.convert_tokens_to_ids(["<bot>"])[0]

    @property
    def human_id(self):
        return self.tokenizer.convert_tokens_to_ids(["<human>"])[0]


    def string2ids(self, string):
        # return self.tokenizer.encode(string)

        # return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string))

        # 注意这里在句首加了空格，但官方表示预训练的时候是没有没有的，因此可能带来结果的下降
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string, add_prefix_space=True))


    def ids2string(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces)

