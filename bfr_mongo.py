#!/usr/bin/env python3
"""
Author: bob.currier@gcoos.org
Create Date:   2017-03-08
Modified Date:   2022-12-02
Notes: MongoDB routines for Guardian. We'll need to convert all the
habscope specific stuff to Guardian/BFR references as we develop
the data structures.
"""
import os
import sys
import json
import time
import logging
import hashlib
import collections
from datetime import datetime
import pymongo as pymongo
from pymongo import MongoClient
#from flask_login import current_user


def load_user(email):
    """
    Let's get loaded!
    """
    client = connect_mongo()
    db = client.bfr
    user = db.users.find({'user_email': email})
    return user


def auth_user(id, pw):
    """
    Who's that knockin' at my door?
    """
    client = connect_mongo()
    db = client.bfr
    try:
        user = db.users.find({'user_email': id.lower()})[0]
    except IndexError:
        return False
    hash = hashlib.md5(pw.encode())
    if hash.hexdigest() == user['user_pw']:
        return True
    else:
        return False


def connect_mongo():
    """
    D'oh
    """
    client = MongoClient('mongo:27017')
    return client


def fetch_record(collection, id):
    """
    gets a single active record
    """
    client = connect_mongo()
    db = client.bfr
    results = db[collection].find({"_id": id})
    for record in results:
        return(record)


def fetch_deleted_record(collection, id):
    """
    gets a single deleted record
    """
    client = connect_mongo()
    db = client.bfr
    results = db[collection].find({"_id": id})[0]
    return results


def fetch_deleted(collection):
    """
    Gets deleted records so they can be counted or undeleted.
    """
    client = connect_mongo()
    db = client.bfr
    the_records = []
    results = (db[collection].find().sort("processed_ts",
                                        pymongo.DESCENDING))
    for result in results:
        processed_time = result['processed_ts']
        processed_time = (datetime.fromtimestamp(
                          processed_time).strftime('%Y-%m-%d %H:%M:%S'))
        result['processed_time'] = processed_time
        the_records.append(result)
    return the_records


def fetch_records(collection, taxa):
    """
    gets all records for given taxa
    """
    client = connect_mongo()
    db = client.bfr
    the_records = []
    if taxa == 'all':
        results = db[collection].find().sort("recorded_ts", pymongo.DESCENDING)
    else:
        results = db[collection].find({"taxa" : taxa}).sort("recorded_ts",
                                                          pymongo.DESCENDING)

    for result in results:
        recorded_ts = result['recorded_ts']
        recorded_ts = (datetime.utcfromtimestamp(
            recorded_ts).strftime('%Y-%m-%d %H:%M:%S'))
        result['recorded_ts'] = recorded_ts
        processed_ts = result['processed_ts']
        processed_ts = (datetime.utcfromtimestamp(
            processed_ts).strftime('%Y-%m-%d %H:%M:%S'))
        result['processed_ts'] = processed_ts
        the_records.append(result)
    return the_records


def insert_record(collection, record_json):
    """
    Log ROIs for later analysis
    """
    logging.debug('insert_record(%s): %s', collection, record_json)
    client = connect_mongo()
    db = client.bfr
    result = db[collection].insert_one(record_json)


def manual_update_record(collection, id, taxa, form):
    """
    Created:    2022-04-27
    Modified:   2022-04-28
    Notes:      Fired from editData.html. Now using taxa
    """
    the_user = load_user(current_user.id.lower())
    analyst = the_user[0]['user_name']
    client = connect_mongo()
    db = client.bfr
    # Status
    db[collection].update({'_id': id},
                        {'$set': {'status': form['status']}})

    # Analyst
    db[collection].update({'_id': id},
                         {'$set': {'analyst': analyst}})

    db[collection].update({'_id': id},
                        {'$set': {'cells_manual': int(form['manCells'])}})

    # mCPL -- here we use the scale to calculate c/L from manCells in form
    m_cpl = calc_cellcount(int(form['manCells']), taxa)
    db[collection].update({'_id': id},
                        {'$set': {'cpl_manual': m_cpl}})

    # GPS
    db[collection].update({'_id': id},
                        {'$set': {'user_gps': [form['posLon'], form['posLat']]}})

    # Location
    db[collection].update({'_id': id},
                        {'$set': {'site': form['location']}})



def build_raw_doc(config, args, wav_file):
    """
    Create raw file document for insertion into MongoDB
    This is the atomic data structure.
    """
    target = args['target']
    processed_dir = config["targets"][target]["processed_dir"]
    pi = config["targets"][target]["pi"]
    project = config["targets"][target]["project"]
    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    annotated_out = '%s/%s_annotated.png' % (processed_dir, no_ext)
    mp4_out = "%s/%s_boosted_sox_processed.mp4" % (processed_dir, no_ext)

    epoch = int(time.time())
    record = {}
    record['processed_epoch'] = epoch
    record['pi'] = pi
    record['project'] = project
    record['raw_file'] = wav_file
    record['mel_file'] = annotated_out 
    record['mp4_file'] = mp4_out

    logging.debug('build_raw_doc(): %s', record)
    return record
