from pandas_dedupe.utility_functions import *

import os
import logging

import dedupe


import pandas as pd




def link_dataframes(dfa, dfb, field_properties, config_name="link_dataframes"):
    
    config_name = config_name.replace(" ", "_")
    
    settings_file = config_name + '_learned_settings'
    training_file = config_name + '_training.json'
 
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

        fields = []
        select_fields(fields, field_properties)
                
              
                
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
