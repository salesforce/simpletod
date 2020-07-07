
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import sys, os
import json
from collections import Counter
import sqlite3
import ipdb
import random
from utils.multiwoz.nlp import normalize, normalize_for_sql
import pprint

import logging
import time

from colorama import Fore, Back, Style

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers.file_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_gpt2").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)




class MultiWozDB(object):
    # loading databases

    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    dbs = {}
    CUR_DIR = os.path.dirname(__file__)

    hotel_info = ['name', 'area', 'internet', 'parking', 'phone', 'postcode', 'pricerange', 'stars', 'takesbookings', 'type', 'address']
    train_info = ["arriveBy", "day", "departure", "destination", "duration", "leaveAt", "price", "trainID"]
    restaurant_info = ["address", "area", "food", "id", "introduction", "name", "phone", "postcode", "pricerange", "signature","type"]
    attraction_info = ["address", "area", "entrance fee", "id", "name", "openhours", "phone", "postcode", "pricerange", "type"]
    taxi_info = []
    database_keys = {
        'hotel': hotel_info,
        'train': train_info,
        'restaurant': restaurant_info,
        'attraction': attraction_info
    }

    for domain in domains:
        db = os.path.join('utils/multiwoz/db/{}-dbase.db'.format(domain))
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs[domain] = c

    def queryResultVenues(self, domain, turn, real_belief=False):
        # query the db

        sql_query = "select {} from {}".format(','.join(self.database_keys[domain]), domain)
        #     sql_query = "select * from {}".format(domain)

        if real_belief == True:
            items = turn.items()
        else:
            items = turn['metadata'][domain]['semi'].items()

        flag = True
        for key, val in items:
            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
                pass
            if 'book' in key:
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize_for_sql(val2)
                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize_for_sql(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction  where name = 'queens college'"
            results = self.dbs[domain].execute(sql_query).fetchall()
            print(sql_query)
            results_dic = []
            for a in results:
                a_dic = dict.fromkeys(self.database_keys[domain])
                for k, v in zip(self.database_keys[domain], a):
                    a_dic[k] = v
                results_dic.append(a_dic)
            print(results_dic)
            return results_dic

        except:
            return []  # TODO test it

    def queryResultVenues_new(self, domain, turn, real_belief=False):
        # query the db
        # sql_query = "select * from {}".format(domain)
        sql_query = "select {} from {}".format(','.join(self.database_keys[domain]), domain)

        if real_belief == True:
            items = turn.items()
        else:
            items = turn['metadata'][domain]['semi'].items()

        flag = True
        for key, val in items:
            if key == 'leaveat':
                key = 'leaveAt'
            if key == 'arriveby':
                key = 'arriveBy'

            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
                pass
            if 'book' in key:
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize_for_sql(val2)

                    # val2 = val2.replace('marys', r"mary's")
                    # val2 = val2.replace('restaurant 17', 'restaurant one seven')
                    # val2 = val2.replace('christ college', r"christ's college")
                    # val2 = val2.replace('city centre north bed and breakfast', 'city centre north b and b')

                    if key == 'name' and val2 in ['the cow pizza kitchen and bar',
                                                  'cow pizza kitchen and bar',
                                                  'wankworth house']:
                        continue


                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize_for_sql(val2)

                    # val2 = val2.replace('marys', r"mary's")
                    # val2 = val2.replace('restaurant 17', 'restaurant one seven')
                    # val2 = val2.replace('christ college', r"christ's college")
                    # val2 = val2.replace('city centre north bed and breakfast', 'city centre north b and b')

                    if key == 'name' and val2 in ['the cow pizza kitchen and bar',
                                                  'cow pizza kitchen and bar',
                                                  'wankworth house']:
                        continue


                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        if ('name', 'restaurant one seven') in list(turn.items()):
            ipdb.set_trace()
        try:  # "select * from attraction  where name = 'queens college'"
            # return self.dbs[domain].execute(sql_query).fetchall()
            results = self.dbs[domain].execute(sql_query).fetchall()
            # print(sql_query)
            results_dic = []
            for a in results:
                a_dic = dict.fromkeys(self.database_keys[domain])
                for k, v in zip(self.database_keys[domain], a):
                    a_dic[k] = v
                results_dic.append(a_dic)
            # print(results_dic)
            return results_dic
        except:
            return []  # TODO test it


def get_belief_new(sent):
    if '<|belief|>' in sent:
        tmp = sent.strip(' ').split('<|belief|>')[-1].split('<|action|>')[0]
    # elif 'belief.' in sent:
    #     tmp = sent.strip(' ').split('<belief>')[-1].split('<action>')[0]
    # elif 'belief' not in sent:
    #     return []
    else:
        return []
    # else:
    #     raise TypeError('unknown belief separator')
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofbelief>')
    tmp = tmp.replace('<|endofbelief|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_belief_new_openaigpt(sent):
    if '< | belief | >' in sent:
        tmp = sent.strip(' ').split('< | belief | >')[-1].split('< | action | >')[0]
    # elif 'belief.' in sent:
    #     tmp = sent.strip(' ').split('<belief>')[-1].split('<action>')[0]
    # elif 'belief' not in sent:
    #     return []
    else:
        return []
    # else:
    #     raise TypeError('unknown belief separator')
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofbelief>')
    tmp = tmp.replace('< | endofbelief | >', '')
    tmp = tmp.replace('<|endoftext|>', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_belief_new_dbsearch(sent):
    if '<|belief|>' in sent:
        tmp = sent.strip(' ').split('<|belief|>')[-1].split('<|endofbelief|>')[0]
    # elif 'belief.' in sent:
    #     tmp = sent.strip(' ').split('<belief>')[-1].split('<action>')[0]
    # elif 'belief' not in sent:
    #     return []
    else:
        return []
    # else:
    #     raise TypeError('unknown belief separator')
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofbelief>')
    tmp = tmp.replace('<|endofbelief|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_action_new_openaigpt(sent):
    if '< | belief | >' in sent:
        tmp = sent.split('< | belief | >')[-1].split('< | response | >')[0].split('< | action | >')[-1].strip()
    elif '< | action | >' in sent:
        tmp = sent.split('< | response | >')[0].split('< | action | >')[-1].strip()
    else:
        return []
    tmp = tmp.strip(' .,')
    # if not tmp.endswith('<endofaction>'):
    #     ipdb.set_trace()
    tmp = tmp.replace('< | endofaction | >', '')
    tmp = tmp.replace('< | endofbelief | >', '')
    tmp = tmp.replace('<|endoftext|>', '')
    action = tmp.split(',')
    new_action = []
    for act in action:
        if act == '':
            continue
        act = act.strip(' .,')
        if act not in new_action:
            act = act.replace('i d', 'id')
            new_action.append(act)
    return new_action


def get_action_new(sent):
    if '<|action|>' not in sent:
        return []
    elif '<|belief|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|response|>')[0].split('<|action|>')[-1].strip()
    elif '<|action|>' in sent:
        tmp = sent.split('<|response|>')[0].split('<|action|>')[-1].strip()
    else:
        return []
    tmp = tmp.strip(' .,')
    # if not tmp.endswith('<endofaction>'):
    #     ipdb.set_trace()
    tmp = tmp.replace('<|endofaction|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    action = tmp.split(',')
    new_action = []
    for act in action:
        if act == '':
            continue
        act = act.strip(' .,')
        if act not in new_action:
            new_action.append(act)
    return new_action


def get_response_new(sent):
    if '<|response|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|action|>')[-1].split('<|response|>')[-1]
    else:
        return ''
    # if '<belief>' in sent:
    #     tmp = sent.split('<belief>')[-1].split('<action>')[-1].split('<response>')[-1]
    # elif '<action>' in sent:
    #     tmp = sent.split('<action>')[-1].split('<response>')[-1]
    # elif '<response>' in sent:
    #     tmp = sent.split('<response>')[-1]
    # else:
    #     tmp = sent
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofresponse>')
    tmp = tmp.replace('<|endofresponse|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    tokens = tokenizer.encode(tmp)
    new_tokens = []
    for tok in tokens:
        # if tok in break_tokens:
        if tok in tokenizer.encode(tokenizer._eos_token):
            continue
        new_tokens.append(tok)
    # ipdb.set_trace()
    response = tokenizer.decode(new_tokens).strip(' ,.')
    return response


def convert_belief(belief):
    dic = {}
    for bs in belief:
        if bs in [' ', '']:
            continue
        domain = bs.split(' ')[0]
        slot = bs.split(' ')[1]
        if slot == 'book':
            slot = ' '.join(bs.split(' ')[1:3])
            value = ' '.join(bs.split(' ')[3:])
        else:
            value = ' '.join(bs.split(' ')[2:])
        if domain not in dic:
            dic[domain] = {}
        try:
            dic[domain][slot] = value
        except:
            print(domain)
            print(slot)
    return dic


def get_db_text(belief_domain, dom, only_match=False):
    db_text_tmp = []
    # for dom in belief_domain:
    if dom not in ['restaurant', 'hotel', 'attraction', 'train']:
        db_text_tmp = ''
    domain_match = len(multiwoz_db.queryResultVenues_new(dom, belief_domain[dom], real_belief=True))

    if dom != 'train':
        if domain_match >= 5:
            domain_match_text = '>=5'
        else:
            domain_match_text = '={}'.format(domain_match)

    elif dom == 'train':
        if domain_match == 0:
            domain_match_text = '=0'
        elif domain_match == 2:
            domain_match_text = '<3'
        elif domain_match == 5:
            domain_match_text = '<6'
        elif domain_match == 10:
            domain_match_text = '<11'
        elif domain_match == 40:
            domain_match_text = '<41'
        else:
            domain_match_text = '>40'

    # if 'fail_book' in goal[dom]:
    #     for item in goal[dom]['fail_book'].items():
    #         if item in belief_book_domain[dom].items():
    #             domain_book_text = 'not available'
    #             break
    #         else:
    #             domain_book_text = 'available'
    # else:
    #     domain_book_text = 'available'
    if domain_match == 0:
        domain_book_text = 'not available'
    else:
        domain_book_text = 'available'


    # if USE_DB_BOOK_DYNAMIC:
    if only_match:
        db_text_tmp.append('{} match{}'.format(dom, domain_match_text))
    else:
        db_text_tmp.append('{} match{} booking={}'.format(dom, domain_match_text, domain_book_text))

    return db_text_tmp


def lexicalize_train(delex_response, db_results, turn_beliefs, turn_domain):
    if len(db_results) > 0:
        sample = random.sample(db_results, k=1)[0]
        value_count = len(db_results)
    else:
        # domain = list(beliefs.keys())[0]
        sample = turn_beliefs[turn_domain]
        value_count = 0

    # print(sample)
    lex_response = delex_response

    if 'from [value_place] to [value_place]' in delex_response:
        departure = sample['departure']
        destination = sample['destination']
        lex_response = lex_response.replace('from [value_place] to [value_place]', 'from {} to {}'.format(departure, destination))
    if 'from [value_place] on [value_day]' in delex_response:
        departure = sample['departure']
        day = sample['day']
        lex_response = lex_response.replace('from [value_place] on [value_day]', 'from {} on {}'.format(departure, day))

    if 'from [value_place]' in delex_response:
        departure = sample['departure']
        # destination = sample['destination']
        lex_response = lex_response.replace('from [value_place]', 'from {}'.format(departure))

    if 'leaving [value_place] at [value_day]' in delex_response:
        departure = sample['departure']
        day = sample['day']
        lex_response = lex_response.replace('leaving [value_place] at [value_day]', 'leaving {} at {}'.format(departure, day))

    if 'leaving [value_place] at [value_time]' in delex_response:
        leaveat = sample['leaveAt']
        departure = sample['departure']
        lex_response = lex_response.replace('leaving [value_place] at [value_time]', 'leaving {} at {}'.format(departure, leaveat))
    if 'leaves [value_place] at [value_time]' in delex_response:
        leaveat = sample['leaveAt']
        departure = sample['departure']
        lex_response = lex_response.replace('leaves [value_place] at [value_time]', 'leaves {} at {}'.format(departure, leaveat))
    if 'leaves at [value_time]' in delex_response:
        if 'leaveAt' in sample:
            leaveat = sample['leaveAt']
            lex_response = lex_response.replace('leaves at [value_time]', 'leaves at {}'.format(leaveat))
    if 'other at [value_time]' in delex_response:
        leaveat = sample['leaveAt']
        lex_response = lex_response.replace('other at [value_time]', 'other at {}'.format(leaveat))

    if 'arrives in [value_place] at [value_time]' in delex_response:
        arriveby = sample['arriveBy']
        destination = sample['destination']
        lex_response = lex_response.replace('arrives in [value_place] at [value_time]', 'arrives in {} at {}'.format(destination, arriveby))
    if 'arrives at [value_time]' in delex_response:
        arriveby = sample['arriveBy']
        lex_response = lex_response.replace('arrives at [value_time]', 'arrives at {}'.format(arriveby))

    if '[value_count] of these' in delex_response:
        value_count = 'one'
        lex_response = lex_response.replace('[value_count] of these', value_count)
    if '[value_count] minutes' in delex_response:
        lex_response = lex_response.replace('[value_count] minutes', sample['duration'])
    if '[value_count]' in delex_response:
        value_count = str(value_count)
        lex_response = lex_response.replace('[value_count]', value_count)
    if 'leaving [value_place]' in delex_response:
        departure = sample['departure']
        lex_response = lex_response.replace('leaving [value_place]', 'leaving {}'.format(departure))
    if 'leaves [value_place]' in delex_response:
        departure = sample['departure']
        lex_response = lex_response.replace('leaves [value_place]', 'leaves {}'.format(departure))
    if 'arrives in [value_place]' in delex_response:
        destination = sample['destination']
        lex_response = lex_response.replace('arrives in [value_place]', 'arrives in {}'.format(destination))
    if '[train_id]' in delex_response:
        train_id = sample['trainID']
        lex_response = lex_response.replace('[train_id]', train_id)
    if '[value_day]' in delex_response:
        train_day = sample['day']
        lex_response = lex_response.replace('[value_day]', train_day)
    if '[value_price]' in delex_response:
        train_price = sample['price']
        lex_response = lex_response.replace('[value_price]', train_price)
    if '[train_reference]' in delex_response:
        random_number = random.randint(10000,99999)
        lex_response = lex_response.replace('[train_reference]', str(random_number))



    return lex_response


def lexicalize_hotel(delex_response, db_results, turn_beliefs, turn_domain):
    if len(db_results) > 0:
        sample = random.sample(db_results, k=1)[0]
        value_count = len(db_results)
    else:
        # ipdb.set_trace()
        # domain = list(beliefs.keys())[0]
        sample = turn_beliefs[turn_domain]
        value_count = 0

    # print(sample)
    lex_response = delex_response
    try:
        if '[hotel_name]' in delex_response:
            lex_response = lex_response.replace('[hotel_name]', sample['name'])
        if '[hotel_address]' in delex_response:
            lex_response = lex_response.replace('[hotel_address]', sample['address'])
        if '[value_area]' in delex_response:
            lex_response = lex_response.replace('[value_area]', sample['area'])
        if 'starting [value_day]' in delex_response:
            lex_response = lex_response.replace('starting [value_day]', 'starting {}'.format(beliefs['book day']))
        if '[value_pricerange]' in delex_response:
            lex_response = lex_response.replace('[value_pricerange]', sample['pricerange'])
        if '[value_count] star' in delex_response:
            lex_response = lex_response.replace('[value_count] star', '{} star'.format(sample['stars']))
        if '[value_count]' in delex_response:
            lex_response = lex_response.replace('[value_count]', str(value_count))
        if '[hotel_reference]' in delex_response:
            random_number = random.randint(10000, 99999)
            lex_response = lex_response.replace('[hotel_reference]', str(random_number))
        if 'starting [value_day]' in delex_response:
            lex_response = lex_response.replace('starting [value_day]', 'starting {}'.format(beliefs['book day']))
        if '[value_count] people' in delex_response:
            lex_response = lex_response.replace('[value_count] people', '{} people'.format(beliefs['book people']))
        if '[value_count] nights' in delex_response:
            lex_response = lex_response.replace('[value_count] nights', '{} nights'.format(beliefs['book stay']))
    except:
        ipdb.set_trace()

    return lex_response


def get_turn_domain_old(b, a):
    tmp = {}
    turn_domain = None
    if a == b:
        turn_domain = list(a.keys())[0]
    # elif len(b.keys()) > len(a.keys()):
    #     turn_domain = list(set(b) - set(a))[0]
    else:
        for domain in b.keys():
            if domain not in a:
                turn_domain = domain
                tmp = b
                break
            tmp = {k: b[domain][k] for k in set(b[domain]) - set(a[domain])}
            if tmp != {}:
                turn_domain = domain
                break
    if not turn_domain:
        ipdb.set_trace()
    print('domain change')
    print('chane', tmp)
    print(b)
    print(a)
    # domain = list(tmp.keys())
    # if len(domain) > 1:
    #     raise TypeError()
    # elif len(domain) == 0:
    #     domain = list(a.keys())[0]
    # else:
    #     domain = domain[0]
    return turn_domain


def get_turn_domain(beliefs, q):
    for k in beliefs.keys():
        if k not in q:
            q.append(k)
            turn_domain = k
            return turn_domain
    return q[-1]


pp = pprint.PrettyPrinter(indent=4)
prev_beliefs = {}
domain_queue = []

if __name__ == '__main__':

    print('\33]0;SimpleTOD\a', end='')
    sys.stdout.flush()

    model_checkpoint = sys.argv[1]
    decoding = sys.argv[2]
    if decoding == 'nucleus':
        TOP_P = float(sys.argv[3])

    delay = 0.5
    multiwoz_db = MultiWozDB()

    print('\nLoading Model', end="")

    if 'openai' in model_checkpoint:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

    # model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.to('cuda')

    break_tokens = tokenizer.encode(tokenizer._eos_token) + tokenizer.encode('?') + tokenizer.encode('!')
    # break_tokens = tokenizer.encode(tokenizer._eos_token)
    MAX_LEN = model.config.n_ctx

    if 'openai-gpt' in model_checkpoint:
        tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

    sample = 1
    print()
    print(Fore.MAGENTA + '\nSimpleTOD is ready to chat. What would you like to ask?' + Style.RESET_ALL)
    # history = []
    context = ''
    input_text = ''
    turn = 0
    # dbmatch = 0

    while True:
        print(Fore.GREEN)
        raw_text = input('You: ')
        print(Style.RESET_ALL)
        input_text = raw_text.replace('you> ', '')
        if input_text in ['q', 'quit']:
            break
        user = '<|user|> {}'.format(input_text)
        context = context + ' ' + user
        text = '<|endoftext|> <|context|> {} <|endofcontext|>'.format(context)

        # print(context)

        text = text.strip()
        indexed_tokens = tokenizer.encode(text)

        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1*MAX_LEN:]
        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = indexed_tokens[-1]

        # if decoding == 'nucleus':
        #     sample_output = model.generate(
        #         tokens_tensor,
        #         do_sample=True,
        #         max_length=MAX_LEN,
        #         top_p=TOP_P,
        #         top_k=0
        #     )
        # elif decoding == 'greedy':
        #     sample_output = model.generate(
        #         tokens_tensor,
        #         max_length=MAX_LEN,
        #         do_sample=False
        #     )
        # predicted_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)


        with torch.no_grad():
            while predicted_index not in break_tokens:
                outputs = model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                if len(indexed_tokens) > MAX_LEN:
                    break
                if tokenizer.decode(indexed_tokens).endswith('<|endofbelief|>'):
                    break

        tmp_pred = tokenizer.decode(indexed_tokens)
        belief_text = get_belief_new_dbsearch(tmp_pred)
        # print(tmp_pred)
        beliefs = convert_belief(belief_text)
        # domain = list(beliefs.keys())[0]
        domain = get_turn_domain(beliefs, domain_queue)

        if 'db' in model_checkpoint:
            if 'dbnmatch' in model_checkpoint:
                only_match = True
                db_text_tmp = get_db_text(beliefs, dom=domain, only_match=only_match)
            else:
                db_text_tmp = get_db_text(beliefs, dom=domain)
            db_text = ' <|dbsearch|> {} <|endofdbsearch|>'.format(' , '.join(db_text_tmp))
            text = tmp_pred + db_text
        # print(text)

        # continue generation after creating db
        indexed_tokens = tokenizer.encode(text)
        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1 * MAX_LEN:]

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = indexed_tokens[-1]

        truncate_action = False
        # Predict all tokens
        with torch.no_grad():
            while predicted_index not in break_tokens:
                outputs = model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                if len(indexed_tokens) > MAX_LEN:
                    break


                predicted_text = tokenizer.decode(indexed_tokens)
                if '<|action|>' in predicted_text:
                    generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[0].split(',')
                    new_actions = []
                    for a in generated_actions:
                        if a in ['', ' ']:
                            continue
                        new_actions.append(a.strip())
                    len_actions = len(new_actions)
                    if len(list(set(new_actions))) > len(new_actions) or (len_actions > 10 and not truncate_action):
                        # ipdb.set_trace()
                        actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                        indexed_tokens = tokenizer.encode('{} {}'.format(predicted_text.split('<|action|>')[0], actions))
                        # print('action truncated')
                        truncate_action = True
                tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

        predicted_text = tokenizer.decode(indexed_tokens)

        action_text = get_action_new(predicted_text)
        response_text = get_response_new(predicted_text)
        # print(predicted_text)

        db_results = multiwoz_db.queryResultVenues_new(domain, beliefs[domain], real_belief=True)

        if domain == 'train':
            lex_response = lexicalize_train(response_text, db_results, beliefs, turn_domain=domain)
        elif domain == 'hotel':
            lex_response = lexicalize_hotel(response_text, db_results, beliefs, turn_domain=domain)
        else:
            ipdb.set_trace()
            raise TypeError('unknown domain')

        delex_system = '<|system|> {}'.format(response_text)
        system = '<|system|> {}'.format(lex_response)
        context = context + ' ' + system


        print(Fore.CYAN + 'SimpleTOD: ', end="")
        for a in lex_response.split(' '):
            print(a + ' ', end="")
            sys.stdout.flush()
            time.sleep(delay)
        print(Style.RESET_ALL)
        print(Fore.YELLOW + 'belief: {}'.format(beliefs) + Style.RESET_ALL)

        print(Style.RESET_ALL)

        turn += 1
        prev_beliefs = beliefs

