

def get_belief(sent):
    if '<|belief|>' in sent:
        tmp = sent.strip(' ').split('<|belief|>')[-1].split('<|action|>')[0]
    else:
        return []
    tmp = tmp.strip(' .,')
    tmp = tmp.replace('<|endofbelief|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_belief_dbsearch(sent):
    if '<|belief|>' in sent:
        tmp = sent.strip(' ').split('<|belief|>')[-1].split('<|endofbelief|>')[0]
    else:
        return []
    tmp = tmp.strip(' .,')
    tmp = tmp.replace('<|endofbelief|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_belief_openaigpt(sent):
    if '< | belief | >' in sent:
        tmp = sent.strip(' ').split('< | belief | >')[-1].split('< | action | >')[0]
    else:
        return []
    tmp = tmp.strip(' .,')
    tmp = tmp.replace('< | endofbelief | >', '')
    tmp = tmp.replace('< | endoftext | >', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_response(sent, tokenizer):
    if '<|response|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|action|>')[-1].split('<|response|>')[-1]
    else:
        return ''
    tmp = tmp.strip(' .,')
    tmp = tmp.replace('<|endofresponse|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    tokens = tokenizer.encode(tmp)
    new_tokens = []
    for tok in tokens:
        if tok in tokenizer.encode(tokenizer._eos_token):
            continue
        new_tokens.append(tok)
    response = tokenizer.decode(new_tokens).strip(' ,.')
    return response


def get_response_openaigpt(sent, tokenizer):
    if '< | response | >' in sent:
        tmp = sent.split('< | belief | >')[-1].split('< | action | >')[-1].split('< | response | >')[-1]
    else:
        return ''
    tmp = tmp.strip(' .,')
    tmp = tmp.replace('< | endofresponse | >', '')
    tmp = tmp.replace('< | endoftext | >', '')
    tokens = tokenizer.encode(tmp)
    new_tokens = []
    for tok in tokens:
        if tok in tokenizer.encode(tokenizer._eos_token):
            continue
        new_tokens.append(tok)
    response = tokenizer.decode(new_tokens).strip(' ,.')
    response = response.replace('[ ', '[')
    response = response.replace(' ]', ']')
    response = response.replace(' _ ', '_')
    response = response.replace('i d', 'id')
    return response


def get_action(sent):
    if '<|action|>' not in sent:
        return []
    elif '<|belief|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|response|>')[0].split('<|action|>')[-1].strip()
    elif '<|action|>' in sent:
        tmp = sent.split('<|response|>')[0].split('<|action|>')[-1].strip()
    else:
        return []
    tmp = tmp.strip(' .,')
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


def get_action_openaigpt(sent):
    if '< | belief | >' in sent:
        tmp = sent.split('< | belief | >')[-1].split('< | response | >')[0].split('< | action | >')[-1].strip()
    elif '< | action | >' in sent:
        tmp = sent.split('< | response | >')[0].split('< | action | >')[-1].strip()
    else:
        return []
    tmp = tmp.strip(' .,')
    tmp = tmp.replace('< | endofaction | >', '')
    tmp = tmp.replace('< | endoftext | >', '')
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


def get_db_dynamically(predicted_text, goal, multiwoz_db):
    gen_belief = get_belief_dbsearch(predicted_text)
    belief_domain = {}
    belief_book_domain = {}
    for bs in gen_belief:
        if bs in ['', ' ']:
            continue
        bs_domain = bs.split()[0]
        if 'book' in bs:
            bs_slot = bs.split()[2]
            bs_val = ' '.join(bs.split()[3:])
            if bs_domain not in belief_book_domain:
                belief_book_domain[bs_domain] = {}
            belief_book_domain[bs_domain][bs_slot] = bs_val
        else:
            bs_slot = bs.split()[1]
            bs_val = ' '.join(bs.split()[2:])
            if bs_domain not in belief_domain:
                belief_domain[bs_domain] = {}
                belief_book_domain[bs_domain] = {}
            belief_domain[bs_domain][bs_slot] = bs_val

    db_text_tmp = []
    for dom in belief_domain:
        if dom not in ['restaurant', 'hotel', 'attraction', 'train']:
            continue
        domain_match = len(multiwoz_db.queryResultVenues(dom, belief_domain[dom], real_belief=True))

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
        if 'fail_book' in goal[dom]:
            for item in goal[dom]['fail_book'].items():
                if item in belief_book_domain[dom].items():
                    domain_book_text = 'not available'
                    break
                else:
                    domain_book_text = 'available'
        else:
            if domain_match == 0:
                domain_book_text = 'not available'
            else:
                domain_book_text = 'available'

        db_text_tmp.append('{} match{} booking={}'.format(dom, domain_match_text, domain_book_text))
    db_text = ' <|dbsearch|> {} <|endofdbsearch|>'.format(' , '.join(db_text_tmp))
    return db_text
