
import argparse
import os
import torch
import logging
from datetime import datetime
import ipdb


class ArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_dir', default='./resources')
        parser.add_argument('--log_dir', default='./logs')
        parser.add_argument('--dataset', default='multiwoz')
        parser.add_argument('--save_dir', default='./checkpoints')

        # generation args for SimpleTOD
        parser.add_argument('--checkpoint', default=None, type=str,
                            help='model checkpoint for generation')
        parser.add_argument('--experiment_name', default=None, help='experiment name')
        parser.add_argument('--split_set', default='test')
        parser.add_argument('--use_db_search', action='store_true', default=False,
                            help='use db search in prompt, should be used with oracle belief')
        parser.add_argument('--use_dynamic_db', action='store_true', default=False,
                            help='compute db search dynamically using generated belief')
        parser.add_argument('--use_oracle_belief', action='store_true', default=False,
                            help='generate with oracle belief in simpleTOD')
        parser.add_argument('--use_oracle_action', action='store_true', default=False,
                            help='generate with oracle action in simpleTOD')
        parser.add_argument('--decoding', default='greedy',
                            help='decoding method for simpletod')

        # dataset args
        parser.add_argument('--name', default='multiwoz')
        parser.add_argument('-b', '--batch_size', type=int, default=2048)

        parser.add_argument('--seq_len', type=int, default=512)
        parser.add_argument('--history_length', type=int, default=5,
                            help='number of turns for context history')
        parser.add_argument('--no_history', action='store_true',
                            help='use current turn only')

        parser.add_argument('--mode', type=str, default='train',
                            choices=['train', 'evaluate', 'generate'], help='mode')

        parser.add_argument('--lexical', action='store_true',
                            help='use lexical data')
        parser.add_argument('--use_knowledge', action='store_true',
                            help='use knowledge')
        parser.add_argument('--use_knowledge_for_decoding', action='store_true',
                            help='use knowledge in decoder attention')
        parser.add_argument('--use_all_knowledge', action='store_true',
                            help='use knowledge')
        parser.add_argument('--knowledge_len', type=int, default=50,
                            help='knowledge length x seq_len')
        parser.add_argument('--use_action', action='store_true',
                            help='use action')
        parser.add_argument('--use_belief', action='store_true',
                            help='use belief')
        parser.add_argument('--use_action_only', action='store_true',
                            help='use action as target only')
        parser.add_argument('--use_belief_only', action='store_true',
                            help='use belief as target only')
        parser.add_argument('--positive_knowledge', action='store_true',
                            help='use only positive knowledge')
        parser.add_argument('--context_knowledge', action='store_true',
                            help='use context knowledge')

        parser.add_argument('--knowledge_matching', type=str, default='entity',
                            help='criterium to find positive knowledge')
        parser.add_argument("--no_cached", action='store_true',
                            help="do not use cached data")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        return args

