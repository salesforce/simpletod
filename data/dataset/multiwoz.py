import os
import json
import random
import torch
import numpy as np
import ipdb
import warnings

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import json
import pickle


# DEFINE special tokens
SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

SEP0_token = 4
SEP1_token = 5
SEP2_token = 6
SEP3_token = 7

SEP4_token = 8
SEP5_token = 9
SEP6_token = 10
SEP7_token = 11


class MultiWozDataset(object):
    def __init__(self, args, split='train', shuffle=True):
        self.args = args
        self.split = split
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        if args.lexical:
            file_path = os.path.join(self.data_dir, '{}_dials_lexicalized.json'.format(split))
        else:
            file_path = os.path.join(self.data_dir, '{}_dials.json'.format(split))

        if args.no_history:
            if args.lexical:
                input_word2index_name = 'input_lang.word2index_lexicalized.json'
                output_word2index_name = 'output_lang.word2index_lexicalized.json'
                input_index2word_name = 'input_lang.index2word_lexicalized.json'
                output_index2word_name = 'output_lang.index2word_lexicalized.json'
            else:
                input_word2index_name = 'input_lang.word2index.json'
                output_word2index_name = 'output_lang.word2index.json'
                input_index2word_name = 'input_lang.index2word.json'
                output_index2word_name = 'output_lang.index2word.json'
        else:
            if args.lexical:
                input_word2index_name = 'history_lang.word2index_lexicalized.json'
                output_word2index_name = 'history_lang.word2index_lexicalized.json'
                input_index2word_name = 'history_lang.index2word_lexicalized.json'
                output_index2word_name = 'history_lang.index2word_lexicalized.json'
            else:
                input_word2index_name = 'history_lang.word2index.json'
                output_word2index_name = 'history_lang.word2index.json'
                input_index2word_name = 'history_lang.index2word.json'
                output_index2word_name = 'history_lang.index2word.json'

        input_word2index_filepath = os.path.join(self.data_dir, input_word2index_name)
        output_word2index_filepath = os.path.join(self.data_dir, output_word2index_name)
        input_index2word_filepath = os.path.join(self.data_dir, input_index2word_name)
        output_index2word_filepath = os.path.join(self.data_dir, output_index2word_name)


        self.dialogues = json.load(open(file_path, 'rt'))
        self.actions = json.load(open('resources/multi-woz/dialogue_acts.json', 'r'))

        self.input_word2index = json.load(open(input_word2index_filepath, 'rt'))
        self.output_word2index = json.load(open(output_word2index_filepath, 'rt'))
        self.input_index2word = json.load(open(input_index2word_filepath, 'rt'))
        self.output_index2word = json.load(open(output_index2word_filepath, 'rt'))

        # special tokens
        self.sos_token = SOS_token
        self.eos_token = EOS_token
        self.unk_token = UNK_token
        self.pad_token = PAD_token

        dial_names = list(self.dialogues.keys())
        if shuffle:
            random.shuffle(dial_names)

        if args.lexical:
            cached_filename = 'resources/cached_data_lexical_{}.pkl'.format(split)
        else:
            cached_filename = 'resources/cached_data_delex_{}.pkl'.format(split)

        
        if os.path.isfile(cached_filename) and not args.no_cached:
            print('loading cached data from {}'.format(cached_filename))
            self.data = pickle.load(open(cached_filename, 'rb'))
        else:
            print('no cached! creating data')
            self.data = {}
            for name in dial_names:
                val_file = self.dialogues[name]
                input_tensor = []
                target_tensor = []
                bs_tensor = []
                db_tensor = []
                input_raw = []
                target_raw = []
                action_raw = []
                belief_raw = []
                for idx, (usr, sys, bs, db, bstate, sys_act) in enumerate(
                        zip(val_file['usr'], val_file['sys'], val_file['bs'], val_file['db'], val_file['bstate'], val_file['sys_act'])):
                    tensor = [self.input_word2index[word] for word in usr.strip(' ').split(' ')] + [
                        EOS_token]
                    input_tensor.append(torch.LongTensor(tensor))  # .view(-1, 1))

                    tensor = [self.output_word2index[word] for word in sys.strip(' ').split(' ')] + [EOS_token]
                    target_tensor.append(torch.LongTensor(tensor))  # .view(-1, 1)

                    input_raw.append(usr)
                    target_raw.append(sys)
                    action_raw.append(sys_act)
                    belief_raw.append(bstate)

                    bs_tensor.append([float(belief) for belief in bs])
                    db_tensor.append([float(pointer) for pointer in db])


                self.data[name] = {
                    'input': input_tensor,
                    'target': target_tensor,
                    'bs': bs_tensor,
                    'db': db_tensor
                }

                self.data[name]['input_raw'] = input_raw
                self.data[name]['target_raw'] = target_raw
                self.data[name]['action_raw'] = action_raw
                self.data[name]['belief_raw'] = belief_raw

            print('caching data to {}'.format(cached_filename))
            with open(cached_filename, 'wb') as f:
                pickle.dump(self.data, f)

    def process_action(self, action):
        concat = []
        if len(action) == 0:
            return None
        for domain, act, slot in action:
            concat.extend([domain, act, slot, '_SEP1'])
        return ' '.join(concat)

    def process_belief_state(self, beliefs):
        concat = []
        if len(beliefs) == 0:
            return None
        for domain, slot, value in beliefs:
            if value == 'not mentioned':
                continue
            concat.extend([domain, slot] + value.strip().split(' ') + ['_SEP0'])
        return ' '.join(concat)


    def __len__(self):
        return len(self.data)

    def _pad_sequence(self, tensor):
        tensor_lengths = np.array([len(sentence) for sentence in tensor])
        if self.args.seq_len:
            longest_sent = self.args.seq_len
        else:
            longest_sent = max(tensor_lengths)
        batch_size = len(tensor)
        padded_tensor = np.ones((batch_size, longest_sent)) * self.pad_token

        # copy over the actual sequences
        for i, x_len in enumerate(tensor_lengths):
            sequence = tensor[i]
            if x_len > longest_sent:
                sequence = sequence[-longest_sent:]
            padded_tensor[i, 0:x_len] = sequence

        return padded_tensor, tensor_lengths

    def _pad_sequence_target_action(self, tensor1, tensor2):
        tensor = []
        for seq1, seq2 in zip(tensor1, tensor2):
            if isinstance(seq1, list):
                tensor.append(seq2)
                continue
            tensor.append(torch.cat((seq1, seq2), 0))
        tensor_lengths = np.array([len(sentence) for sentence in tensor])
        if self.args.seq_len:
            longest_sent = self.args.seq_len
        else:
            longest_sent = max(tensor_lengths)
        batch_size = len(tensor)
        padded_tensor = np.ones((batch_size, longest_sent)) * self.pad_token

        # copy over the actual sequences
        for i, x_len in enumerate(tensor_lengths):
            sequence = tensor[i]
            if x_len > longest_sent:
                sequence = sequence[-longest_sent:]
            padded_tensor[i, 0:x_len] = sequence

        return padded_tensor, tensor_lengths

    def _pad_sequence_target_action_belief(self, tensor1, tensor2, tensor3):
        tensor = []
        for seq1, seq2, seq3 in zip(tensor1, tensor2, tensor3):
            if len(seq1) > 0 and len(seq2) > 0:
                tensor.append(torch.cat((seq1, seq2, seq3), 0))
            elif len(seq1) == 0 and len(seq2) == 0:
                tensor.append(seq3)
            elif len(seq1) == 0:
                tensor.append(torch.cat((seq2, seq3), 0))
            elif len(seq2) == 0:
                tensor.append(torch.cat((seq1, seq3), 0))

        tensor_lengths = np.array([len(sentence) for sentence in tensor])
        if self.args.seq_len:
            longest_sent = self.args.seq_len
        else:
            longest_sent = max(tensor_lengths)
        batch_size = len(tensor)
        padded_tensor = np.ones((batch_size, longest_sent)) * self.pad_token

        # copy over the actual sequences
        for i, x_len in enumerate(tensor_lengths):
            sequence = tensor[i]
            if x_len > longest_sent:
                sequence = sequence[-longest_sent:]
            padded_tensor[i, 0:x_len] = sequence

        return padded_tensor, tensor_lengths

    def pad_dialogue(self, dial, name):
        input = dial['input']
        target = dial['target']
        if max([h.shape[0] for h in input]) > self.args.seq_len:
            warnings.warn('input length bigger than max sequence length')
            return None
        padded_input, input_length = self._pad_sequence(input)
        padded_target, target_length = self._pad_sequence(target)
        ret_dial = {'input': padded_input,
                    'input_length': input_length,
                    'target': padded_target,
                    'target_length': target_length,
                    'bs': np.array(dial['bs']),
                    'db': np.array(dial['db']),
                    }
        return ret_dial

    def pad_dialogue_with_history(self, dial, name):
        input = dial['input']
        target = dial['target']
        history = []
        history_pairs = []
        for i, (inp, tgt) in enumerate(zip(input, target)):
            tmp = torch.cat((inp, tgt))
            history_pairs.append(tmp)

        for i, (inp, tgt) in enumerate(zip(input, target)):
            if i == 0:
                hist = inp
            elif i > self.args.history_length:
                hist = torch.cat((*history_pairs[-i:i], inp))
            else:
                hist = torch.cat((*history_pairs[:i], inp))
            history.append(hist)

        padded_input, input_length = self._pad_sequence(input)
        padded_target, target_length = self._pad_sequence(target)
        padded_history, history_length = self._pad_sequence(history)

        ret_dial = {'input': padded_input,
                    'input_length': input_length,
                    'target': padded_target,
                    'target_length': target_length,
                    'history': padded_history,
                    'history_length': history_length,
                    'bs': np.array(dial['bs']),
                    'db': np.array(dial['db']),
                    'input_raw': dial['input_raw'],
                    'target_raw': dial['target_raw'],
                    'action_raw': dial['action_raw'],
                    'belief_raw': dial['belief_raw']
                    }

        return ret_dial

    def __getitem__(self, item):
        ret_dial_name = list(self.data.keys())[item]
        if self.args.no_history:
            dial = self.pad_dialogue(self.data[ret_dial_name], ret_dial_name)
        else:
            dial = self.pad_dialogue_with_history(self.data[ret_dial_name], ret_dial_name)
        dial['name'] = ret_dial_name
        return dial
        


