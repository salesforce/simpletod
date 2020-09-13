import json
from sklearn.metrics import f1_score, accuracy_score
import ipdb
import sys
import numpy as np
from utils.Constants import SLOT_VALS
from utils.dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--eval_file', default=str,
                    help='evaluate file name (json)')
parser.add_argument('--default_cleaning', action='store_true',
                    help='use default cleaning from multiwoz')
parser.add_argument('--type2_cleaning', action='store_true',
                    help='use type 2 cleaning, refer to [https://arxiv.org/abs/2005.00796]')

data = json.load(open(args.eval_file, 'r'))

num_turns = 0
joint_acc = 0

clean_tokens = ['<|endoftext|>']

for dial in data:
    dialogue_pred = data[dial]['generated_turn_belief']
    dialogue_target = data[dial]['target_turn_belief']
    model_context = data[dial]['model_context']

    for turn_id, (turn_target, turn_pred, turn_context) in enumerate(
            zip(dialogue_target, dialogue_pred, model_context)):

        # clean
        for bs in turn_pred:
            if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                turn_pred.remove(bs)

        new_turn_pred = []
        for bs in turn_pred:
            for tok in clean_tokens:
                bs = bs.replace(tok, '').strip()
                new_turn_pred.append(bs)
        turn_pred = new_turn_pred

        turn_pred, turn_target = ignore_none(turn_pred, turn_target)

        # MultiWOZ default cleaning
        if args.default_cleaning:
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

        join_flag = False
        if set(turn_target) == set(turn_pred):
            joint_acc += 1
            join_flag = True
        
        elif args.type2_cleaning: # check for possible Type 2 noisy annotations
            flag = True
            for bs in turn_target:
                if bs not in turn_pred:
                    flag = False
                    break
            if flag:
                for bs in turn_pred:
                    if bs not in dialogue_target_final:
                        flag = False
                        break

            if flag: # model prediction might be correct if found in Type 2 list of noisy annotations
                dial_name = dial.split('.')[0]
                if dial_name in IGNORE_TURNS_TYPE2 and turn_id in IGNORE_TURNS_TYPE2[dial_name]: # ignore these turns
                    pass
                else:
                    joint_acc += 1
                    join_flag = True

        num_turns += 1

joint_acc /= num_turns

print('joint accuracy: {}'.format(joint_acc))

