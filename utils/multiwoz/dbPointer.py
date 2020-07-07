import sqlite3

import numpy as np

from .nlp import normalize
import os

PATH = './utils/multiwoz'

# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']#, 'police']
dbs = {}
for domain in domains:
    db = os.path.join(PATH, 'db/{}-dbase.db'.format(domain))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c


def convert_dbpointer_to_text(vect, goal, belief):
    domain_in_pointer = ['restaurant', 'hotel', 'attraction', 'train']
    restaurant_book_vec = vect[24:26]
    hotel_book_vec = vect[26:28]
    train_book_vec = vect[28:]
    text = []
    for idx in range(4):
        domain = domains[idx]
        if domain not in goal:
            continue
        Flag = False
        for bs in belief:
            if bs[0] == domain:
                Flag = True
        if not Flag: # not bstate for domain
            continue
        domain_vec = vect[idx * 6: idx * 6 + 6]
        if domain != 'train':
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 1
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 3
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 4
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 5
            else:
                raise ValueError('invalid domain match')

            if domain_match >= 5:
                domain_match_text = '>=5'
            else:
                domain_match_text = '={}'.format(domain_match)
            if (domain == 'restaurant' and np.all(restaurant_book_vec == np.array([0, 1]))) or (domain == 'hotel' and np.all(hotel_book_vec == np.array([0, 1]))):
                text.append('{} match{} booking=available'.format(domain, domain_match_text))
            else:
                text.append('{} match{} booking=not available'.format(domain, domain_match_text))

        else: # train domain
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 5
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 10
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 40
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 41
            else:
                raise ValueError('invalid domain match')

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

            if np.all(train_book_vec == np.array([0, 1])):
                text.append('{} match{} booking=available'.format(domain, domain_match_text))
            else:
                text.append('{} match{} booking=not available'.format(domain, domain_match_text))

    return ' , '.join(text)


def convert_dbpointer_to_text_nmatch(vect, goal, belief):
    domain_in_pointer = ['restaurant', 'hotel', 'attraction', 'train']
    restaurant_book_vec = vect[24:26]
    hotel_book_vec = vect[26:28]
    train_book_vec = vect[28:]
    text = []
    for idx in range(4):
        domain = domains[idx]
        if domain not in goal:
            continue
        Flag = False
        for bs in belief:
            if bs[0] == domain:
                Flag = True
        if not Flag: # not bstate for domain
            continue
        domain_vec = vect[idx * 6: idx * 6 + 6]
        if domain != 'train':
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 1
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 3
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 4
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 5
            else:
                raise ValueError('invalid domain match')

            if domain_match >= 5:
                domain_match_text = '>=5'
            else:
                domain_match_text = '={}'.format(domain_match)

            text.append('{} match{}'.format(domain, domain_match_text))

            # if (domain == 'restaurant' and np.all(restaurant_book_vec == np.array([0, 1]))) or (domain == 'hotel' and np.all(hotel_book_vec == np.array([0, 1]))):
            #     # text.append('{} match{} booking=available'.format(domain, domain_match_text))
            #     text.append('{} match{}'.format(domain, domain_match_text))
            # else:
            #     text.append('{} match{} booking=not available'.format(domain, domain_match_text))

        else: # train domain
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 5
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 10
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 40
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 41
            else:
                raise ValueError('invalid domain match')

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

            text.append('{} match{}'.format(domain, domain_match_text))

            # if np.all(train_book_vec == np.array([0, 1])):
            #     text.append('{} match{} booking=available'.format(domain, domain_match_text))
            # else:
            #     text.append('{} match{} booking=not available'.format(domain, domain_match_text))

    return ' , '.join(text)


def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0,0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector


def queryResult(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    #print turn['metadata'][domain]['semi']
    for key, val in turn['metadata'][domain]['semi'].items():
        if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                # change query for trains
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    #try:  # "select * from attraction  where name = 'queens college'"
    #print sql_query
    #print domain
    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities


def queryResultVenues(domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)

    if real_belief == True:
        items = turn.items()
    elif real_belief=='tracking':
        for slot in turn[domain]:
            key = slot[0].split("-")[1]
            val = slot[0].split("-")[2]
            if key == "price range":
                key = "pricerange"
            elif key == "leave at":
                key = "leaveAt"
            elif key == "arrive by":
                key = "arriveBy"
            if val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
        pass
    else:
        items = turn['metadata'][domain]['semi'].items()

    flag = True
    for key, val in items:
        if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " +key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:  # "select * from attraction  where name = 'queens college'"
        return dbs[domain].execute(sql_query).fetchall()
    except:
        return []  # TODO test it
