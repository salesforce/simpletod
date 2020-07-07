# -*- coding: utf-8 -*-
import copy
import json
import os
import sys
import re
import shutil
import urllib
from urllib import request
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm

import numpy as np

from utils.multiwoz import dbPointer
from utils.multiwoz import delexicalize

from utils.multiwoz.nlp import normalize, normalize_lexical, normalize_beliefstate, normalize_mine
import ipdb

np.set_printoptions(precision=3)

np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 1000000
MAX_LENGTH = 600

DATA_DIR = './resources'

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return data

    # if not isinstance(turn, str) and not isinstance(turn, unicode):
    if not isinstance(turn, bytes) and not isinstance(turn, str):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")

    return data


def delexicaliseReferenceNumber(sent, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    if turn['metadata']:
        for domain in domains:
            if turn['metadata'][domain]['book']['booked']:
                for slot in turn['metadata'][domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'
                    key = normalize(turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def delexicaliseReferenceNumber_mine(sent, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    if turn['metadata']:
        for domain in domains:
            if turn['metadata'][domain]['book']['booked']:
                for slot in turn['metadata'][domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'
                    key = normalize_mine(turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize_mine("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize_mine("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def addBookingPointer(task, turn, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if task['goal']['restaurant']:
        # if turn['metadata']['restaurant'].has_key("book"):
        # if turn['metadata']['restaurant']['book'].has_key("booked"):
        if "book" in turn['metadata']['restaurant']:
            if "booked" in turn['metadata']['restaurant']['book']:
                if turn['metadata']['restaurant']['book']["booked"]:
                    if "reference" in turn['metadata']['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if task['goal']['hotel']:
        # if turn['metadata']['hotel'].has_key("book"):
        #     if turn['metadata']['hotel']['book'].has_key("booked"):
        if "book" in turn['metadata']['hotel']:
            if "booked" in turn['metadata']['hotel']['book']:
                if turn['metadata']['hotel']['book']["booked"]:
                    if "reference" in turn['metadata']['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if task['goal']['train']:
        # if turn['metadata']['train'].has_key("book"):
        #     if turn['metadata']['train']['book'].has_key("booked"):
        if "book" in turn['metadata']['train']:
            if "booked" in turn['metadata']['train']['book']:
                if turn['metadata']['train']['book']["booked"]:
                    if "reference" in turn['metadata']['train']['book']["booked"][0]:
                        train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector


def addDBPointer(turn):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = dbPointer.queryResult(domain, turn)
        pointer_vector = dbPointer.oneHotVector(num_entities, domain, pointer_vector)

    return pointer_vector


def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    summary_bstate = []
    for domain in domains:
        domain_active = False

        booking = []
        # print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if bstate[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or \
                    bstate[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    return summary_bstate


def get_belief_state(bstate):
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    raw_bstate = []
    for domain in domains:
        for slot, value in bstate[domain]['semi'].items():
            if value:
                raw_bstate.append((domain, slot, normalize_beliefstate(value)))
        for slot, value in bstate[domain]['book'].items():
            if slot == 'booked':
                continue
            if value:
                new_slot = '{} {}'.format('book', slot)
                raw_bstate.append((domain, new_slot, normalize_beliefstate(value)))
    # ipdb.set_trace()
    return raw_bstate


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        # print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            if 'db_pointer' not in d['log'][i]:
                print('no db')
                return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=True)
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=False)
            belief_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = belief_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def analyze_dialogue_raw_beliefstate(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        # print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            if 'db_pointer' not in d['log'][i]:
                print('no db')
                return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=True)
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=False)
            belief_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = belief_summary
            # get raw belief state
            belief_state = get_belief_state(d['log'][i]['metadata'])
            d['log'][i]['belief_state'] = belief_state
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def analyze_dialogue_raw_beliefstate_v2(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        # print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            if 'db_pointer' not in d['log'][i]:
                print('no db')
                return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=True)
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=False)
            belief_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = belief_summary
            # get raw belief state
            belief_state = get_belief_state(d['log'][i]['metadata'])
            d['log'][i]['belief_state'] = belief_state
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    # d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    d_orig = analyze_dialogue_raw_beliefstate(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    db = [t['db_pointer'] for t in d_orig['usr_log']]
    bs = [t['belief_summary'] for t in d_orig['sys_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    for u, d, s, b in zip(usr, db, sys, bs):
        dial.append((u, s, d, b))

    return dial


def get_dial_raw_bstate(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue_raw_beliefstate(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    db = [t['db_pointer'] for t in d_orig['usr_log']]
    bs = [t['belief_summary'] for t in d_orig['sys_log']]
    belief_state = [t['belief_state'] for t in d_orig['sys_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    for u, d, s, b, bstate in zip(usr, db, sys, bs, belief_state):
        dial.append((u, s, d, b, bstate))

    return dial


def createDict(word_freqs):
    words = [k for k in word_freqs.keys()]
    freqs = [v for v in word_freqs.values()]

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    # Extra vocabulary symbols
    _GO = '_GO'
    EOS = '_EOS'
    UNK = '_UNK'
    PAD = '_PAD'
    SEP0 = '_SEP0'
    SEP1 = '_SEP1'
    SEP2 = '_SEP2'
    SEP3 = '_SEP3'
    SEP4 = '_SEP4'
    SEP5 = '_SEP5'
    SEP6 = '_SEP6'
    SEP7 = '_SEP7'
    extra_tokens = [_GO, EOS, UNK, PAD, SEP0, SEP1, SEP2, SEP3, SEP4, SEP5, SEP6, SEP7]
    # extra_tokens = [_GO, EOS, UNK, PAD]

    worddict = OrderedDict()
    for ii, ww in enumerate(extra_tokens):
        worddict[ww] = ii
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii #+ len(extra_tokens)

    new_worddict = worddict.copy()
    for key, idx in worddict.items():
        if idx >= DICT_SIZE:
            del new_worddict[key]
    return new_worddict


def moveFiles(src_path, dst_path):
    shutil.copy(os.path.join(src_path, 'data.json'), dst_path)
    shutil.copy(os.path.join(src_path, 'valListFile.json'), dst_path)
    shutil.copy(os.path.join(src_path, 'testListFile.json'), dst_path)
    shutil.copy(os.path.join(src_path, 'dialogue_acts.json'), dst_path)
    return


def createDelexData():
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """
    # download the data
    loadDataMultiWoz()

    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    fin1 = open(os.path.join(DATA_DIR, 'multi-woz/data.json'))
    data = json.load(fin1)

    fin2 = open(os.path.join(DATA_DIR, 'multi-woz/dialogue_acts.json'))
    data2 = json.load(fin2)

    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]
        # print dialogue_name

        idx_acts = 1
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])

            words = sent.split()
            sent = delexicalize.delexicalise(' '.join(words), dic)

            # parsing reference number GIVEN belief state
            sent = delexicaliseReferenceNumber(sent, turn)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            sent = re.sub(digitpat, '[value_count]', sent)

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = sent

            if idx % 2 == 1:  # if it's a system turn
                # add database pointer
                pointer_vector = addDBPointer(turn)
                # add booking pointer
                pointer_vector = addBookingPointer(dialogue, turn, pointer_vector)

                # print pointer_vector
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)
            idx_acts += 1

        delex_data[dialogue_name] = dialogue

    with open(os.path.join(DATA_DIR, 'multi-woz/delex.json'), 'w') as outfile:
        json.dump(delex_data, outfile)

    return delex_data


def loadDataMultiWoz():
    data_url = os.path.join(DATA_DIR, 'multi-woz-2.1/data.json')
    dataset_url = "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y"
    download_path = os.path.join(DATA_DIR, 'multi-woz')
    extract_path = os.path.join(download_path, 'MULTIWOZ2.1')
    os.makedirs(download_path, exist_ok=True)

    if not os.path.exists(data_url):
        print("Downloading and unzipping the MultiWOZ dataset")
        resp = urllib.request.urlopen(dataset_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall(download_path)
        zip_ref.close()

        moveFiles(src_path=extract_path, dst_path=download_path)
        return


def createDelexData_mine():
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """
    # download the data
    loadDataMultiWoz()

    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = delexicalize.prepareSlotValuesIndependent_mine()
    delex_data = {}

    fin1 = open(os.path.join(DATA_DIR, 'multi-woz/data.json'))
    data = json.load(fin1)

    fin2 = open(os.path.join(DATA_DIR, 'multi-woz/dialogue_acts.json'))
    data2 = json.load(fin2)

    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]
        # print dialogue_name

        idx_acts = 1
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize_mine(turn['text'])

            # only delexicalize system response
            if idx % 2 == 1:
                words = sent.split()
                sent = delexicalize.delexicalise(' '.join(words), dic)

            # parsing reference number GIVEN belief state
            sent = delexicaliseReferenceNumber_mine(sent, turn)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            # sent = re.sub(digitpat, '[value_count]', sent)

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = sent

            if idx % 2 == 1:  # if it's a system turn
                # add database pointer
                pointer_vector = addDBPointer(turn)
                # add booking pointer
                pointer_vector = addBookingPointer(dialogue, turn, pointer_vector)

                # print pointer_vector
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)
            idx_acts += 1

        delex_data[dialogue_name] = dialogue

    with open(os.path.join(DATA_DIR, 'multi-woz/delex_mine.json'), 'w') as outfile:
        json.dump(delex_data, outfile)

    return delex_data


def createLexicalData():
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """
    # download the data
    loadDataMultiWoz()

    # create dictionary of delexicalied values that then we will search against, order matters here!
    # dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    fin1 = open(os.path.join(DATA_DIR, 'multi-woz/data.json'))
    data = json.load(fin1)

    fin2 = open(os.path.join(DATA_DIR, 'multi-woz/dialogue_acts.json'))
    data2 = json.load(fin2)

    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]
        # print dialogue_name

        idx_acts = 1
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize_lexical(turn['text'])

            # words = sent.split()
            # sent = delexicalize.delexicalise(' '.join(words), dic)

            # parsing reference number GIVEN belief state
            # sent = delexicaliseReferenceNumber(sent, turn)

            # changes to numbers only here
            # digitpat = re.compile('\d+')
            # sent = re.sub(digitpat, '[value_count]', sent)

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = sent

            if idx % 2 == 1:  # if it's a system turn
                # add database pointer
                pointer_vector = addDBPointer(turn)
                # add booking pointer
                pointer_vector = addBookingPointer(dialogue, turn, pointer_vector)

                # print pointer_vector
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

            # FIXING delexicalization:
            # dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)
            idx_acts += 1

        # ipdb.set_trace()
        delex_data[dialogue_name] = dialogue

    with open(os.path.join(DATA_DIR, 'multi-woz/lex.json'), 'w') as outfile:
        json.dump(delex_data, outfile)

    return delex_data


def get_action(actions, dial_name, turn_id):
    turn_id = str(turn_id)
    if turn_id in actions[dial_name.split('.')[0]]:
        turn_action = actions[dial_name.split('.')[0]][turn_id]
        if isinstance(turn_action, str):
            return turn_action, []
        acts = {}
        for k, v in turn_action.items():
            domain, act = [w.lower() for w in k.split('-')]
            for (slot, value) in v:
                slot = ' '.join(slot.lower().strip().split('\t'))
                value = ' '.join(value.lower().strip().split('\t'))
                # concat.extend(['_SEP1', v1, '_SEP2', v2])
                if domain in acts and act in acts[domain] and slot in acts[domain][act]: # already domain-act is considered, skip
                    continue
                if domain not in acts:
                    acts[domain] = {}
                    acts[domain][act] = {}
                    acts[domain][act] = [(slot, value)]
                elif act not in acts[domain]:
                    acts[domain][act] = {}
                    acts[domain][act] = [(slot, value)]
                else:
                    acts[domain][act].append((slot, value))

        concat = []
        for domain in acts:
            for act in acts[domain]:
                for slot, value in acts[domain][act]:
                    concat.append((domain, act, slot))
        return turn_action, concat
    else:
        return [], []


def divideData(data, lexicalize=False):
    """Given test and validation sets, divide
    the data for three different sets"""
    # ipdb.set_trace()
    testListFile = []
    fin = open(os.path.join(DATA_DIR, 'multi-woz/testListFile.json'))
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open(os.path.join(DATA_DIR, 'multi-woz/valListFile.json'))
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    trainListFile = open(os.path.join(DATA_DIR, 'multi-woz/trainListFile'), 'w')

    actions = json.load(open('resources/multi-woz/dialogue_acts.json', 'r'))
    test_dials = {}
    val_dials = {}
    train_dials = {}

    # dictionaries
    word_freqs_usr = OrderedDict()
    word_freqs_sys = OrderedDict()
    word_freqs_history = OrderedDict()
    word_freqs_action = OrderedDict()
    word_freqs_belief = OrderedDict()

    for dialogue_name in tqdm(data):

        dial = get_dial_raw_bstate(data[dialogue_name])

        if dial:
            dialogue = {}
            dialogue['usr'] = []
            dialogue['sys'] = []
            dialogue['db'] = []
            dialogue['bs'] = []
            dialogue['bstate'] = []
            dialogue['sys_act_raw'] = []
            dialogue['sys_act'] = []
            for turn_id, turn in enumerate(dial):
                dialogue['usr'].append(turn[0])
                dialogue['sys'].append(turn[1])
                dialogue['db'].append(turn[2])
                dialogue['bs'].append(turn[3])
                dialogue['bstate'].append(turn[4])

                turn_act_raw, turn_act = get_action(actions, dialogue_name, turn_id+1)
                # ipdb.set_trace()
                dialogue['sys_act_raw'].append(turn_act_raw)
                dialogue['sys_act'].append(turn_act)

            if dialogue_name in testListFile:
                test_dials[dialogue_name] = dialogue
            elif dialogue_name in valListFile:
                val_dials[dialogue_name] = dialogue
            else:
                trainListFile.write(dialogue_name + '\n')
                train_dials[dialogue_name] = dialogue

            for turn in dial:
                line = turn[0]
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs_usr:
                        word_freqs_usr[w] = 0
                    word_freqs_usr[w] += 1

                # dialogue history vocab
                for w in words_in:
                    if w not in word_freqs_history:
                        word_freqs_history[w] = 0
                    word_freqs_history[w] += 1

                line = turn[1]
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs_sys:
                        word_freqs_sys[w] = 0
                    word_freqs_sys[w] += 1

                # dialogue history vocab
                for w in words_in:
                    if w not in word_freqs_history:
                        word_freqs_history[w] = 0
                    word_freqs_history[w] += 1

            act_words = []
            for dial_act in dialogue['sys_act']:
                for domain, act, slot in dial_act:
                    act_words.extend([domain, act, slot])
            for w in act_words:
                if w not in word_freqs_sys:
                    word_freqs_sys[w] = 0
                word_freqs_sys[w] += 1
                if w not in word_freqs_history:
                    word_freqs_history[w] = 0
                word_freqs_history[w] += 1
                if w not in word_freqs_action:
                    word_freqs_action[w] = 0
                word_freqs_action[w] += 1

            belief_words = []
            for dial_bstate in dialogue['bstate']:
                for domain, slot, value in dial_bstate:
                    belief_words.extend([domain, slot])
                    belief_words.extend(normalize_beliefstate(value).strip().split(' '))
            for w in belief_words:
                if w not in word_freqs_sys:
                    word_freqs_sys[w] = 0
                word_freqs_sys[w] += 1
                if w not in word_freqs_history:
                    word_freqs_history[w] = 0
                word_freqs_history[w] += 1
                if w not in word_freqs_belief:
                    word_freqs_belief[w] = 0
                word_freqs_belief[w] += 1

    # save all dialogues
    if lexicalize:
        val_filename = os.path.join(DATA_DIR, 'val_dials_lexicalized.json')
        test_filename = os.path.join(DATA_DIR, 'test_dials_lexicalized.json')
        train_filename = os.path.join(DATA_DIR, 'train_dials_lexicalized.json')
    else:
        val_filename = os.path.join(DATA_DIR, 'val_dials.json')
        test_filename = os.path.join(DATA_DIR, 'test_dials.json')
        train_filename = os.path.join(DATA_DIR, 'train_dials.json')
    
    with open(val_filename, 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open(test_filename, 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open(train_filename, 'w') as f:
        json.dump(train_dials, f, indent=4)

    return word_freqs_usr, word_freqs_sys, word_freqs_history


def buildDictionaries(word_freqs_usr, word_freqs_sys, word_freqs_histoy, lexicalize=False):
    """Build dictionaries for both user and system sides.
    You can specify the size of the dictionary through DICT_SIZE variable."""
    dicts = []
    worddict_usr = createDict(word_freqs_usr)
    dicts.append(worddict_usr)
    worddict_sys = createDict(word_freqs_sys)
    dicts.append(worddict_sys)
    worddict_history = createDict(word_freqs_histoy)
    dicts.append(worddict_history)

    # reverse dictionaries
    idx2words = []
    for dictionary in dicts:
        dic = {}
        for k, v in dictionary.items():
            dic[v] = k
        idx2words.append(dic)

    if lexicalize:
        input_index2word_filename = os.path.join(DATA_DIR, 'input_lang.index2word_lexicalized.json')
        input_word2index_filename = os.path.join(DATA_DIR, 'input_lang.word2index_lexicalized.json')
        output_index2word_filename = os.path.join(DATA_DIR, 'output_lang.index2word_lexicalized.json')
        output_word2index_filename = os.path.join(DATA_DIR, 'output_lang.word2index_lexicalized.json')
        history_index2word_filename = os.path.join(DATA_DIR, 'history_lang.index2word_lexicalized.json')
        history_word2index_filename = os.path.join(DATA_DIR, 'history_lang.word2index_lexicalized.json')
    else:
        input_index2word_filename = os.path.join(DATA_DIR, 'input_lang.index2word.json')
        input_word2index_filename = os.path.join(DATA_DIR, 'input_lang.word2index.json')
        output_index2word_filename = os.path.join(DATA_DIR, 'output_lang.index2word.json')
        output_word2index_filename = os.path.join(DATA_DIR, 'output_lang.word2index.json')
        history_index2word_filename = os.path.join(DATA_DIR, 'history_lang.index2word.json')
        history_word2index_filename = os.path.join(DATA_DIR, 'history_lang.word2index.json')

    with open(input_index2word_filename, 'w') as f:
        json.dump(idx2words[0], f, indent=2)
    with open(input_word2index_filename, 'w') as f:
        json.dump(dicts[0], f, indent=2)
    with open(output_index2word_filename, 'w') as f:
        json.dump(idx2words[1], f, indent=2)
    with open(output_word2index_filename, 'w') as f:
        json.dump(dicts[1], f, indent=2)
    with open(history_index2word_filename, 'w') as f:
        json.dump(idx2words[2], f, indent=2)
    with open(history_word2index_filename, 'w') as f:
        json.dump(dicts[2], f, indent=2)


def main():
    if sys.argv[1] == 'delex':
        print('MultiWoz Create delexicalized dialogues. Get yourself a coffee, this might take a while.')
        if not os.path.isfile(os.path.join(DATA_DIR, 'multi-woz/delex.json')):
            data = createDelexData()
        else:
            data = json.load(open(os.path.join(DATA_DIR, 'multi-woz/delex.json')))
    elif sys.argv[1] == 'lexical':
        print('MultiWoz Create lexicalized dialogues. Get yourself a coffee, this might take a while.')
        if not os.path.isfile(os.path.join(DATA_DIR, 'multi-woz/lex.json')):
            data = createLexicalData()
        else:
            data = json.load(open(os.path.join(DATA_DIR, 'multi-woz/lex.json')))

    else:
        raise TypeError('unknown preprocessing')
    print('Divide dialogues for separate bits - usr, sys, db, bs')
    word_freqs_usr, word_freqs_sys, word_freqs_history = divideData(data,
                                                                    lexicalize=(str(sys.argv[1])=='lexical'))

    print('Building dictionaries')
    buildDictionaries(word_freqs_usr, word_freqs_sys, word_freqs_history,
                      lexicalize=(str(sys.argv[1])=='lexical'))


if __name__ == "__main__":
    main()
