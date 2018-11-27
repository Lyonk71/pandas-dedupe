import os
import logging

import dedupe
from unidecode import unidecode

import pandas as pd


logging.getLogger().setLevel(logging.WARNING)


def trim(x):
    x = x.split()
    x = ' '.join(x)
    return x   


def clean_punctuation(df):
    for i in df.columns:
        df[i] = df[i].astype(str)
    for i in df.columns:
        df[i] = df[i].replace({'$':''}, regex=True)    
    df = df.applymap(lambda x: x.lower())
    for i in df.columns:
        df[i] = df[i].str.replace('[^\w\s\.\\\\]','')
    df = df.applymap(lambda x: trim(x))
    df = df.applymap(lambda x: unidecode(x))
    for i in df.columns:
        df[i] = df[i].replace({'nan': None})
    return df

def select_fields(fields, field_properties):
    for i in field_properties:
        if type(i)==str:
            fields.append({'field': i, 'type': 'String'})
        if len(i)==2:
            fields.append({'field': i[0], 'type': i[1]})
        if len(i)==3:
            if i[2] == 'has missing':
                fields.append({'field': i[0], 'type': i[1], 'has missing': True})
            elif i[2] == 'crf':
                fields.append({'field': i[0], 'type': i[1], 'crf': True})
            else:
                raise Exception(i[2] + " is not a valid field property")
    
def specify_type(df, field_properties):
    for i in field_properties:
        if i[1] == 'Price':
            df[i[0]] = df[i[0]].astype(float)
    
    
def dedupe_dataframe(df, field_properties):
    # Import Data
    
    settings_file = 'csv_example_learned_settings'
    training_file = 'csv_example_training.json'

    print('importing data ...')

    df = clean_punctuation(df)
    
    specify_type(df, field_properties)                
    
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
        
        fields = []
        select_fields(fields, field_properties)


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


def link_dataframes(dfa, dfb, field_properties):
    
    settings_file = 'data_matching_learned_settings'
    training_file = 'data_matching_training.json'
 
    print('importing data ...')

    dfa = clean_punctuation(dfa)
    specify_type(dfa, field_properties)
    
    dfa['index_field'] = dfa.index
    dfa['index_field'] = dfa['index_field'].apply(lambda x: "dfa" + str(x))
    dfa.set_index(['index_field'], inplace=True)
            
    data_1 = dfa.to_dict(orient='index')
   

    dfb = clean_punctuation(dfb)
    specify_type(dfb, field_properties)
    
    dfb['index_field'] = dfb.index
    dfb['index_field'] = dfb['index_field'].apply(lambda x: "dfb" + str(x))
    dfb.set_index(['index_field'], inplace=True)

    
    data_2 = dfb.to_dict(orient='index')
    # ---------------------------------------------------------------------------------



    # ## Training


    if os.path.exists(settings_file):
        print('reading from', settings_file)
        with open(settings_file, 'rb') as sf :
            linker = dedupe.StaticRecordLink(sf)

    else:
        # Define the fields the linker will pay attention to
        #
        # Notice how we are telling the linker to use a custom field comparator
        # for the 'price' field. 
        fields =[]

        for i in field_properties:
            if type(i)==str:
                fields.append({'field': i, 'type': 'String'})
            if len(i)==2:
                fields.append({'field': i[0], 'type': i[1]})
            if len(i)==3:
                fields.append({'field': i[0], 'type': i[1], 'has missing': True})
                
              
                
        # Create a new linker object and pass our data model to it.
        linker = dedupe.RecordLink(fields)
        # To train the linker, we feed it a sample of records.
        linker.sample(data_1, data_2, 15000)

        # If we have training data saved from a previous run of linker,
        # look for it an load it in.
        # __Note:__ if you want to train from scratch, delete the training_file
        if os.path.exists(training_file):
            print('reading labeled examples from ', training_file)
            with open(training_file) as tf :
                linker.readTraining(tf)

        # ## Active learning
        # Dedupe will find the next pair of records
        # it is least certain about and ask you to label them as matches
        # or not.
        # use 'y', 'n' and 'u' keys to flag duplicates
        # press 'f' when you are finished
        print('starting active labeling...')

        dedupe.consoleLabel(linker)
        linker.train()

        # When finished, save our training away to disk
        with open(training_file, 'w') as tf :
            linker.writeTraining(tf)

        # Save our weights and predicates to disk.  If the settings file
        # exists, we will skip all the training and learning next time we run
        # this file.
        with open(settings_file, 'wb') as sf :
            linker.writeSettings(sf)


    # ## Blocking

    # ## Clustering

    # Find the threshold that will maximize a weighted average of our
    # precision and recall.  When we set the recall weight to 2, we are
    # saying we care twice as much about recall as we do precision.
    #
    # If we had more data, we would not pass in all the blocked data into
    # this function but a representative sample.

    print('clustering...')
    linked_records = linker.match(data_1, data_2, 0)

    print('# duplicate sets', len(linked_records))
    

    #Convert linked records into dataframe
    df_linked_records = pd.DataFrame(linked_records)
    
    df_linked_records['dfa_link'] = df_linked_records[0].apply(lambda x: x[0])
    df_linked_records['dfb_link'] = df_linked_records[0].apply(lambda x: x[1])
    df_linked_records.rename(columns={1: 'confidence'}, inplace=True)
    df_linked_records.drop(columns=[0], inplace=True)
    df_linked_records['cluster id'] = df_linked_records.index

   
    #For both dfa & dfb, add cluster id & confidence score from liked_records
    dfa.index.rename('dfa_link', inplace=True)
    dfa = dfa.merge(df_linked_records, on='dfa_link', how='left')

    dfb.index.rename('dfb_link', inplace=True)
    dfb = dfb.merge(df_linked_records, on='dfb_link', how='left')

    #Concatenate results from dfa + dfb
    df_final = dfa.append(dfb, ignore_index=True, sort=True)
    df_final = df_final.sort_values(by=['cluster id'])
    df_final = df_final.drop(columns=['dfa_link','dfb_link'])

    return df_final
