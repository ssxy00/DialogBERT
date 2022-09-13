# DialogBERT
# Copyright 2021-present NAVER Corp.
# BSD 3-clause

import os
import logging
import torch

from models import DialogBERT
from data_loader import DialogTransformerDataset, HBertMseEuopDataset
from learner import Learner

logger = logging.getLogger(__name__)

    
def get_optim_params(models, args):
    no_decay = ['bias', 'LayerNorm.weight']
    parameters = []
    for model in models:
        parameters.append(
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': args.weight_decay})
        parameters.append(
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0})
    return parameters


class DialogBERTSolver(object):
    def __init__(self, args, model=None):
        self.model = model    
        self.build(args)
        
    def build(self, args):
        # Load pretrained model and tokenizer
        if self.model is None:
            self.model = DialogBERT(args)
        self.model.to(args.device)

    def load(self, args):
        # Load a trained model and vocabulary that you have fine-tuned
        assert args.reload_path, "please specify the checkpoint path in args.reload_path"
        self.model.from_pretrained(args.reload_path)
        self.model.to(args.device)
        
    def train(self, args):   
        
        ## Train All
        # ssxy: there are special settings for MUR and DUOR, but GLCM can ignore
        train_set = HBertMseEuopDataset(
            os.path.join(args.data_path, 'train.h5'), 
            self.model.tokenizer, 
            context_shuf=True, context_masklm=True
        )
        valid_set = HBertMseEuopDataset(os.path.join(args.data_path, 'valid.h5'), self.model.tokenizer)
        test_set = HBertMseEuopDataset(os.path.join(args.data_path, 'test.h5'), self.model.tokenizer)

        optim_params = get_optim_params([self.model], args)
        global_step, tr_loss = Learner().run_train(
            args, self.model, train_set, optim_params, entry='forward', valid_set=valid_set)
        
        return global_step, tr_loss
    
    def evaluate(self, args):
        self.load(args)
        test_set = HBertMseEuopDataset(os.path.join(args.data_path, 'test.h5'), self.model.tokenizer)
        result, generated_text = Learner().run_eval(args, self.model, test_set)
        with open(args.eval_output_path, 'w') as f_eval:
            f_eval.write(generated_text+'\n')
        return result    
 