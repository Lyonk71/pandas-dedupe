from pandas_dedupe.utility_functions import (
    clean_punctuation,
    select_fields,
    specify_type
)

import os
import logging
import math

import dedupe

import pandas as pd


logging.getLogger().setLevel(logging.WARNING)


def _train(settings_file, training_file, data, field_properties, sample_size):
    """Internal method that trains the deduper model if a training file does
    not exist.

        Parameters
        ----------
        settings_file : str
            A path to a settings file that will be loaded if it exists.
        training_file : str
            A path to a training file that will be loaded to keep training
            from.
        data : dict
            The dictionary form of the dataframe that dedupe requires.
        field_properties : dict
            The mapping of fields to their respective data types. Please
            see the dedupe documentation for further details.
        sample_size : float, default 0.3
            Specify the sample size used for training as a float from 0 to 1.
            By default it is 30% (0.3) of our data.

        Returns
        -------
        dedupe.Dedupe
            A dedupe model instance.
    """
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
        sample_num = math.floor(len(data) * sample_size)
        deduper.sample(data, sample_num)

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

    return deduper


def _cluster(deduper, data, threshold, canonicalize):
    """Internal method that clusters the data.

        Parameters
        ----------
        deduper : dedupe.Deduper
            A trained instance of dedupe.
        data : dict
            The dedupe formatted data dictionary.
        threshold : dedupe.Threshold
            The threshold used for clustering.
        canonicalize : bool or list, default False
            Option that provides the canonical records as additional columns.
            Specifying a list of column names only canonicalizes those columns.

        Returns
        -------
        pd.DataFrame
            A dataframe storing the clustering results.
    """
    # ## Clustering
    print('clustering...')
    clustered_dupes = deduper.match(data, threshold)

    print('# duplicate sets', len(clustered_dupes))

    # Convert data_d to string so that Price & LatLong won't get traceback
    # during dedupe.canonicalize()
    for i in data.values():
        for key in i:
            if i[key] is None:
                pass
            else:
                i[key] = str(i[key])
    
    df_data = []
    # ## Writing Results
    cluster_id = 0
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        id_set, scores = cluster
        cluster_d = [data[c] for c in id_set]

        canonical_rep = None
        if canonicalize:
            canonical_rep = dedupe.canonicalize(cluster_d)

        for record_id, score in zip(id_set, scores):
            tmp = {
                'Id': record_id,
                'cluster id': cluster_id,
                'confidence': score,
            }

            if canonicalize:
                fields_to_canon = canonical_rep.keys()

                if isinstance(canonicalize, list):
                    fields_to_canon = canonicalize

                for key in fields_to_canon:
                    canon_key = 'canonical_' + key
                    tmp[canon_key] = canonical_rep[key]

            df_data.append(tmp)

    clustered_df = pd.DataFrame(df_data)
    clustered_df = clustered_df.set_index('Id')

    return clustered_df


def dedupe_dataframe(df, field_properties, canonicalize=False,
                     config_name="dedupe_dataframe", recall_weight=1,
                     sample_size=0.3):
    """Deduplicates a dataframe given fields of interest.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to deduplicate.
        field_properties : list
            A list specifying what fields to use for deduplicating records.
        canonicalize : bool or list, default False
            Option that provides the canonical records as additional columns.
            Specifying a list of column names only canonicalizes those columns.
        config_name : str, default dedupe_dataframe
            The configuration file name. Note that this will be used as 
            a prefix to save the settings and training files.
        recall_weight : int, default 1
            Find the threshold that will maximize a weighted average of our
            precision and recall.  When we set the recall weight to 2, we are
            saying we care twice as much about recall as we do precision.
        sample_size : float, default 0.3
            Specify the sample size used for training as a float from 0 to 1.
            By default it is 30% (0.3) of our data.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe that contains the cluster id and confidence
            score. Optionally, it will contain canonicalized columns for all
            attributes of the record.

    """
    # Import Data  
    config_name = config_name.replace(" ", "_")
   
    settings_file = config_name + '_learned_settings'
    training_file = config_name + '_training.json'

    print('importing data ...')

    df = clean_punctuation(df)
    
    specify_type(df, field_properties)                
    
    df['dictionary'] = df.apply(
        lambda x: dict(zip(df.columns, x.tolist())), axis=1)
    data_d = dict(zip(df.index, df.dictionary))
    
    # train or load the model
    deduper = _train(settings_file, training_file, data_d, field_properties,
                     sample_size)

    # ## Set threshold
    threshold = deduper.threshold(data_d, recall_weight=recall_weight)

    # cluster the records
    clustered_df = _cluster(deduper, data_d, threshold, canonicalize)
    results = df.join(clustered_df, how='left')
    results.drop(['dictionary'], axis=1, inplace=True)

    return results