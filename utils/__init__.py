
import os
from tensorboardX import SummaryWriter
from datetime import datetime
import copy
import torch
import logging
logging.basicConfig()

import ipdb


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def get_writer(name):
    writer = SummaryWriter(name)
    return writer


def prepare_logger_writer(args):
    # if args.resume or args.mode in ['evaluate', 'generate']:
    #     chekpoint_name = os.path.join(args.experiment_dir, '{}_best.chkpt'.format(args.model_type))
    #     chekpoint_params = torch.load(chekpoint_name)
    #     config = chekpoint_params['config']
    #     args.name = config.name
    #     args.log_dir = config.log_dir
    #     args.exp_name = config.exp_name
    args.logger = get_logger(args.name)
    args.writer = get_writer(os.path.join(args.log_dir, args.exp_name))
    return args


def get_config(args):
    config = copy.copy(args)
    config.writer = None
    return config
