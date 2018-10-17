from future.builtins import next

import os
import csv
import re
import logging
import optparse

import dedupe
from unidecode import unidecode

import pandas as pd


logging.getLogger().setLevel(logging.WARNING)

def trim(x):
    x = x.split()
    x = ' '.join(x)
    return x   

settings_file = 'csv_example_learned_settings'
training_file = 'csv_example_training.json'


def deduplicate(df, field_properties):
    # Import Data

    print('importing data ...')

    df = df.astype(str, inplace=True)
    df = df.applymap(lambda x: x.lower())
    for i in df.columns:
        df[i] = df[i].str.replace('[^\w\s]','')
    df = df.applymap(lambda x: trim(x))
    dfa = dfa.applymap(lambda x: unidecode(x))

    df['dictionary'] = df.apply(lambda x: dict(zip(df.columns,x.tolist())), axis=1)
    data_d = dict(zip(df.index,df.dictionary))

    # If a settings file already exists, we'll just load that and skip training
    if os.path.exists(settings_file):
        print('reading from', settings_file)
        with open(settings_file, 'rb') as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        # ## Training

        # Define the fields dedupe will pay attention to
        fields =[]

        for i in field_properties:
            if type(i)==str:
                fields.append({'field': i, 'type': 'String'})
            if len(i)==2:
                fields.append({'field': i[0], 'type': i[1]})

        # Create a new deduper object and pass our data model to it.
        deduper = dedupe.Dedupe(fields)

        # To train dedupe, we feed it a sample of records.
        deduper.sample(data_d, 15000)

        # If we have training data saved from a previous run of dedupe,
        # look for it and load it in.
        # __Note:__ if you want to train from scratch, delete the training_file
        if os.path.exists(training_file):
            print('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.readTraining(f)


        print('starting active labeling...')

        dedupe.consoleLabel(deduper)

        # Using the examples we just labeled, train the deduper and learn
        # blocking predicates
        deduper.train()

        # When finished, save our training to disk
        with open(training_file, 'w') as tf:
            deduper.writeTraining(tf)

        # Save our weights and predicates to disk.  If the settings file
        # exists, we will skip all the training and learning next time we run
        # this file.
        with open(settings_file, 'wb') as sf:
            deduper.writeSettings(sf)

    # ## Set threshold

    threshold = deduper.threshold(data_d, recall_weight=1)

    # ## Clustering

    print('clustering...')
    clustered_dupes = deduper.match(data_d, threshold)

    print('# duplicate sets', len(clustered_dupes))

    # ## Writing Results

    # turn results into dataframe


    cluster_membership = {}
    cluster_id = 0
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        id_set, scores = cluster
        cluster_d = [data_d[c] for c in id_set]
        canonical_rep = dedupe.canonicalize(cluster_d)
        for record_id, score in zip(id_set, scores):
            cluster_membership[record_id] = {
                "cluster id" : cluster_id,
                "canonical representation" : canonical_rep,
                "confidence": score
            }



    cluster_index=[]
    for i in cluster_membership.items():
        cluster_index.append(i)


    dfa = pd.DataFrame(cluster_index)

    dfa.rename(columns={0: 'Id'}, inplace=True)

    dfa['cluster id'] = dfa[1].apply(lambda x: x["cluster id"])
    dfa['confidence'] = dfa[1].apply(lambda x: x["confidence"])

    for i in dfa[1][0]['canonical representation'].keys():
        dfa[i + ' - ' + 'canonical'] = None
        dfa[i + ' - ' + 'canonical'] = dfa[1].apply(lambda x: x['canonical representation'][i])



    dfa.set_index('Id', inplace=True)

    df = df.join(dfa)

    df.drop(columns=[1, 'dictionary'], inplace=True)

    return df
