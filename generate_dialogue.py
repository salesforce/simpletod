
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
from evaluate_multiwoz import MultiWozDB
from utils.multiwoz import dbPointer
from utils.simpletod import *
import tqdm
import json
import ipdb
import sys
import os



opt = ArgsParser().parse()
opt.multiwoz_version = '2.1'
opt.use_action = True
opt.use_knowledge = True
opt.context_knowledge = True
opt.lexical = True



HISTORY_LEN = None
# USE_ORACLE_BELIEF = True
USE_ORACLE_BELIEF = opt.use_oracle_belief
# USE_ORACLE_ACTION = False
USE_ORACLE_ACTION = opt.use_oracle_action
# USE_DB_SEARCH = True
USE_DB_SEARCH = opt.use_db_search
USE_DYNAMIC_DB = opt.use_dynamic_db
# EVAL_SPLIT = 'test'
EVAL_SPLIT = opt.split_set

decoding = opt.decoding

multiwoz_data = json.load(open('resources/multi-woz/lex.json','r'))

# provide model name and checkpoint directory
# exp_name = 'gpt2'
# exp_name = opt.experiment_name
# checkpoint = opt.checkpoint
# model_checkpoint = '../dialog-transformer/output/{}/{}/'.format(exp_name, checkpoint)
model_checkpoint = opt.checkpoint
exp_name = os.path.split(model_checkpoint)[0].split('/')[-2]

multiwoz_db = MultiWozDB()

opt_delex = ArgsParser().parse()
opt_delex.multiwoz_version = '2.1'


data = MultiWozDataset(opt, split=EVAL_SPLIT, shuffle=False)

data_delex = MultiWozDataset(opt_delex, split=EVAL_SPLIT, shuffle=False)

lex_dict = {}
delex_dict = {}
for d in data:
    lex_dict[d['name']] = d

for d in data_delex:
    delex_dict[d['name']] = d

if 'openai-gpt' in model_checkpoint:
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)

if 'openai-gpt' in model_checkpoint:
    model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
else:
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

model.eval()
model.to('cuda')

break_tokens = tokenizer.encode(tokenizer._eos_token)
MAX_LEN = model.config.n_ctx


generated_dict = {}
num_data = len(data)

for i, dial_name in enumerate(lex_dict):
    if EVAL_SPLIT == 'train' and i > 1000:
        break
    d = lex_dict[dial_name]
    d_delex = delex_dict[dial_name]
    print('{} [{}/{}] \r'.format(d['name'], i, num_data), end='')
    sys.stdout.flush()
    beliefs_raw = d['belief_raw']
    user = d['input_raw']
    system = d['target_raw']
    system_delex = d_delex['target_raw']
    if 'delex' in model_checkpoint:
        target_response = system_delex
    else:
        target_response = system


    action = d['action_raw']
    target_action = []
    for turn_act in action:
        turn_action = []
        for act in turn_act:
            act_str = '{} {} {}'.format(act[0], act[1], act[2])
            turn_action.append(act_str)
        target_action.append(turn_action)

    dialogue_aggregated_target_belief = []
    dialogue_target_belief = []

    for turn_belief in beliefs_raw:
        turn_belief_str = []
        for bs in turn_belief:
            domain, slot, value = bs
            if value in ['not mentioned', 'none']:
                continue
            bs_str = '{} {} {}'.format(domain.lower(), slot.lower(), value.lower())
            if bs_str not in dialogue_aggregated_target_belief:
                dialogue_aggregated_target_belief.append(bs_str)
            turn_belief_str.append(bs_str)
        dialogue_target_belief.append(turn_belief_str)


    db_data = d['db']
    goal = multiwoz_data[dial_name]['goal']

    generated = []
    model_context = []
    for turn_id, (usr_turn, _) in enumerate(zip(user, system)):

        if turn_id == 0:
            tmp_text = '<|user|> {}'.format(usr_turn.strip())
        else:
            tmp = []
            for k in range(turn_id):
                tmp.append('<|user|> {}'.format(user[k].strip()))
                tmp.append('<|system|> {}'.format(system[k].strip()))

            tmp.append('<|user|> {}'.format(usr_turn.strip()))

            # trim history
            if HISTORY_LEN and len(tmp) > HISTORY_LEN:
                tmp = tmp[-1*HISTORY_LEN:]
            tmp_text = ' '.join(tmp)
        
        if dial_name == 'SNG02319.json':
            tmp_text = tmp_text.replace('300 will', '03:00 will')

        text = '{} <|context|> {} <|endofcontext|> '.format(tokenizer._bos_token, tmp_text)

        if USE_ORACLE_BELIEF:
            turn_belief = dialogue_target_belief[turn_id]
            belief_str = '<|belief|> {} <|endofbelief|>'.format(' , '.join(turn_belief))
            text = text + ' ' + belief_str
        
        db_text = dbPointer.convert_dbpointer_to_text(db_data[turn_id], goal, beliefs_raw[turn_id])
        if USE_DB_SEARCH and USE_ORACLE_BELIEF:
            if not USE_ORACLE_BELIEF:
                print('warning: oracle db is true, oracle belief is false')
            text += ' <|dbsearch|> {} <|endofdbsearch|>'.format(db_text)
        
        if USE_ORACLE_ACTION:
            turn_action = target_action[turn_id]
            action_str = '<|action|> {} <|endofaction|>'.format(' , '.join(turn_action))
            text = text + ' ' + action_str


        model_context.append(text)
        indexed_tokens = tokenizer.encode(text)
        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1*MAX_LEN:]

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = indexed_tokens[-1]

        if USE_DB_SEARCH and not USE_ORACLE_BELIEF: # generate belief, then get DB search results, then continue generation (greedy decoding)
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
            if not USE_DYNAMIC_DB: # use oracle db
                text = '{} {}'.format(tmp_pred, db_text)
            else: # use dynamic db search results (using generated belief)
                db_text_dynamic = get_db_dynamically(tmp_pred, goal, multiwoz_db=multiwoz_db)
                text = '{} {}'.format(tmp_pred, db_text_dynamic)

            # continue generation
            indexed_tokens = tokenizer.encode(text)
            if len(indexed_tokens) > MAX_LEN:
                indexed_tokens = indexed_tokens[-1 * MAX_LEN:]

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])

            # If you have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to('cuda')
            predicted_index = indexed_tokens[-1]

            # Predict all tokens
            with torch.no_grad():
                #while predicted_index not in break_tokens:
                while predicted_index not in break_tokens:
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]
                    predicted_index = torch.argmax(predictions[0, -1, :]).item()
                    indexed_tokens += [predicted_index]

                    # sometime model generate repeated actions, we just use truncate actions if this happens
                    predicted_text = tokenizer.decode(indexed_tokens)
                    if '<|action|>' in predicted_text:
                        generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[
                            0].split(',')
                        new_actions = []
                        for a in generated_actions:
                            if a in ['', ' ']:
                                continue
                            new_actions.append(a.strip())
                        len_actions = len(new_actions)
                        if len(list(set(new_actions))) > len(new_actions) or (
                                len_actions > 10 and not truncate_action):
                            actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                            indexed_tokens = tokenizer.encode(
                                '{} {}'.format(predicted_text.split('<|action|>')[0], actions))
                            truncate_action = True

                    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                    if len(indexed_tokens) > MAX_LEN:
                        break
                    if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                        break

                predicted_text = tokenizer.decode(indexed_tokens)
                generated.append(predicted_text)

        else: # generate belief, action, and response once
            with torch.no_grad():

                if decoding == 'nucleus':
                    sample_output = model.generate(
                        tokens_tensor,
                        # indexed_tokens,
                        do_sample=True,
                        max_length=MAX_LEN,
                        top_p=0.5,
                        top_k=0
                    )
                    predicted_text = tokenizer.decode(sample_output[0])
                    tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                    predicted_text = tmp
                    generated.append(predicted_text)

                elif decoding == 'greedy':
                    # GREEDY DECODING
                    
                    # sample_output = model.generate(
                    #     # tokens_tensor,
                    #     indexed_tokens,
                    #     max_length=MAX_LEN,
                    #     do_sample=False
                    # )
                    
                    while predicted_index not in break_tokens:
                        outputs = model(tokens_tensor)
                        predictions = outputs[0]
                        predicted_index = torch.argmax(predictions[0, -1, :]).item()
                        indexed_tokens += [predicted_index]
                        
                        # sometime model generate repeated actions, we just use truncate actions if this happens
                        predicted_text = tokenizer.decode(indexed_tokens)
                        if '<|action|>' in predicted_text:
                            generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[
                                0].split(',')
                            new_actions = []
                            for a in generated_actions:
                                if a in ['', ' ']:
                                    continue
                                new_actions.append(a.strip())
                            len_actions = len(new_actions)
                            if len(list(set(new_actions))) > len(new_actions) or (
                                    len_actions > 10 and not truncate_action):
                                actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                                indexed_tokens = tokenizer.encode(
                                    '{} {}'.format(predicted_text.split('<|action|>')[0], actions))
                                truncate_action = True
                                
                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                        if len(indexed_tokens) > MAX_LEN:
                            break
                        if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                            break
                            
                    predicted_text = tokenizer.decode(indexed_tokens)
                    tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                    generated.append(predicted_text)

                # predicted_text = tokenizer.decode(sample_output[0])
                # tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                # predicted_text = tmp
                # generated.append(predicted_text)

    dialogue_aggregated_pred_belief = []
    dialogue_pred_belief = []
    dialogue_pred_responses = []
    dialogue_pred_action = []

    # aggregate belief states
    for turn, pred in enumerate(generated):
        turn_pred_belief = []
        if 'openai-gpt' in model_checkpoint:
            belief = get_belief_openaigpt(pred)
        else:
            if 'dbsearch' in model_checkpoint or 'dbnmatch' in model_checkpoint or USE_DB_SEARCH or 'db' in model_checkpoint:
                belief = get_belief_dbsearch(pred)
            else:
                belief = get_belief(pred)
        if len(belief) > 0:
            for bs in belief:
                if bs not in ['', ' '] and bs not in dialogue_aggregated_pred_belief:
                    dialogue_aggregated_pred_belief.append(bs)
            new_belief = list(set(belief))
            dialogue_pred_belief.append(new_belief)
        else:
            if len(dialogue_pred_belief) == 0:
                dialogue_pred_belief.append([''])
            else:
                dialogue_pred_belief.append(dialogue_pred_belief[-1])
        if 'openai-gpt' in model_checkpoint:
            gen_response = get_response_openaigpt(pred, tokenizer)
        else:
            gen_response = get_response(pred, tokenizer)
        dialogue_pred_responses.append(gen_response)

        if 'openai-gpt' in model_checkpoint:
            gen_action = get_action_openaigpt(pred)
        else:
            gen_action = get_action(pred)
        dialogue_pred_action.append(gen_action)

    generated_dict[d['name']] = {
        'target_belief': dialogue_aggregated_target_belief,
        'target_turn_belief': dialogue_target_belief,
        'generated_belief': dialogue_aggregated_pred_belief,
        'generated_turn_belief': dialogue_pred_belief,
        'target_response': target_response,
        'generated_response': dialogue_pred_responses,
        'target_action': target_action,
        'generated_action': dialogue_pred_action,
        'target_user': user,
        'model_context': model_context
    }


save_name = '{}_{}'.format(exp_name, EVAL_SPLIT)

if USE_ORACLE_BELIEF:
    save_name += '_oracleBelief'

if USE_DB_SEARCH:
    save_name += '_oracleDB'

if USE_ORACLE_ACTION:
    save_name += '_oracleAction'

if HISTORY_LEN:
    save_name += '_context[history={}]'.format(HISTORY_LEN)
else:
    save_name += '_context[history=full_history]'

save_name += '_nocarry'

with open('{}.json'.format(save_name), 'wt') as f:
    json.dump(generated_dict, f)
