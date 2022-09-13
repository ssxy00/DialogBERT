# DialogBERT
# Copyright 2021-present NAVER Corp.
# BSD 3-clause

# coding=utf-8
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from tqdm import tqdm
import numpy as np
import torch

import models, solvers, data_loader

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", default='/home1/sxy/DialogBERT/datasets/', type=str, help="The input data path.")
    parser.add_argument("--output_path", default='/home1/sxy/DialogBERT/output', type=str, help="The output data path.")
    parser.add_argument("--gpt2_vocab_dir", default="/home1/sxy/models/transformers3_gpt2-small", type=str,
                        help="bert-base-uncased path")
    parser.add_argument("--dataset", default='multiwoz', type=str, help="dataset name")
    parser.add_argument("--model", default="DialogBERT", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--n_epochs", default=1.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, help="random seed for initialization")

    # for test
    parser.add_argument('--do_test', action='store_true', help="whether test or train")
    parser.add_argument("--reload_path", type=str, help="path to load optimal checkpoint.")
    parser.add_argument("--eval_output_path", default='/home1/sxy/DialogBERT/output/tmp.txt',
                        type=str, help="The output data path.")

    args = parser.parse_args()

    args.data_path = os.path.join(args.data_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset, f"lr{args.learning_rate}")

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(args)

    solver = getattr(solvers, args.model + 'Solver')(args)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if not args.do_test:
        global_step, tr_loss = solver.train(args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    else:
        # Evaluation
        results = solver.evaluate(args)
        print(results)


if __name__ == "__main__":
    main()
