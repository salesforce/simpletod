

GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports",
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall",
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east",
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre",
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
        # day
        "next friday":"friday", "monda": "monday",
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",
        }


IGNORE_TURNS_TYPE2 = \
    {
        'PMUL1812': [1, 2],
        'MUL2177': [9, 10, 11, 12, 13, 14],
        'PMUL0182': [1],
        'PMUL0095': [1],
        'MUL1883': [1],
        'PMUL2869': [9, 11],
        'SNG0433': [0, 1],
        'PMUL4880': [2],
        'PMUL2452': [2],
        'PMUL2882': [2],
        'SNG01391': [1],
        'MUL0803': [7],
        'MUL1560': [4, 5],
        'PMUL4964': [6, 7],
        'MUL1753': [8, 9],
        'PMUL3921': [4],
        'PMUL3403': [0, 4],
        'SNG0933': [3],
        'SNG0296': [1],
        'SNG0477': [1],
        'MUL0814': [1],
        'SNG0078': [1],
        'PMUL1036': [6],
        'PMUL4840': [2],
        'PMUL3423': [6],
        'MUL2284': [2],
        'PMUL1373': [1],
        'SNG01538': [1],
        'MUL0011': [2],
        'PMUL4326': [4],
        'MUL1697': [10],
        'MUL0014': [5],
        'PMUL1370': [1],
        'PMUL1801': [7],
        'MUL0466': [2],
        'PMUL0506': [1, 2],
        'SNG1036': [2]

}



def ignore_none(pred_belief, target_belief):
    for pred in pred_belief:
        if 'catherine s' in pred:
            pred.replace('catherine s', 'catherines')

    clean_target_belief = []
    clean_pred_belief = []
    for bs in target_belief:
        if 'not mentioned' in bs:
            continue
        clean_target_belief.append(bs)

    for bs in pred_belief:
        if 'not mentioned' in bs:
            continue
        clean_pred_belief.append(bs)

    target_belief = clean_target_belief
    pred_belief = clean_pred_belief

    return pred_belief, target_belief


def fix_mismatch_jason(slot, value):
    # miss match slot and value
    if slot == "type" and value in ["nigh", "moderate -ly priced", "bed and breakfast",
                                  "centre", "venetian", "intern", "a cheap -er hotel"] or \
            slot == "internet" and value == "4" or \
            slot == "pricerange" and value == "2" or \
            slot == "type" and value in ["gastropub", "la raza", "galleria", "gallery",
                                       "science", "m"] or \
            "area" in slot and value in ["moderate"] or \
            "day" in slot and value == "t":
        value = "none"
    elif slot == "type" and value in ["hotel with free parking and free wifi", "4",
                                    "3 star hotel"]:
        value = "hotel"
    elif slot == "star" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no":
            value = "north"
        elif value == "we":
            value = "west"
        elif value == "cent":
            value = "centre"
    elif "day" in slot:
        if value == "we":
            value = "wednesday"
        elif value == "no":
            value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if slot == "area" and value in ["stansted airport", "cambridge", "silver street"] or \
            slot == "area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
        value = "none"
    return slot, value


def default_cleaning(pred_belief, target_belief):
    pred_belief_jason = []
    target_belief_jason = []
    for pred in pred_belief:
        if pred in ['', ' ']:
            continue
        domain = pred.split()[0]
        if 'book' in pred:
            slot = ' '.join(pred.split()[1:3])
            val = ' '.join(pred.split()[3:])
        else:
            slot = pred.split()[1]
            val = ' '.join(pred.split()[2:])

        if slot in GENERAL_TYPO:
            val = GENERAL_TYPO[slot]

        slot, val = fix_mismatch_jason(slot, val)

        pred_belief_jason.append('{} {} {}'.format(domain, slot, val))

    for tgt in target_belief:
        domain = tgt.split()[0]
        if 'book' in tgt:
            slot = ' '.join(tgt.split()[1:3])
            val = ' '.join(tgt.split()[3:])
        else:
            slot = tgt.split()[1]
            val = ' '.join(tgt.split()[2:])

        if slot in GENERAL_TYPO:
            val = GENERAL_TYPO[slot]
        slot, val = fix_mismatch_jason(slot, val)
        target_belief_jason.append('{} {} {}'.format(domain, slot, val))

    turn_pred = pred_belief_jason
    turn_target = target_belief_jason

    return turn_pred, turn_target