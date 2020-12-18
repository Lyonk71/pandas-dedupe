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


def _active_learning(data, sample_size, deduper, training_file, settings_file):
    """Internal method that trains the deduper model using active learning.
        Parameters
        ----------
        data : dict
            The dictionary form of the dataframe that dedupe requires.
        sample_size : float, default 0.3
            Specify the sample size used for training as a float from 0 to 1.
            By default it is 30% (0.3) of our data.
        deduper : a dedupe model instance
        training_file : str
            A path to a training file that will be loaded to keep training
            from.
        settings_file : str
            A path to a settings file that will be loaded if it exists.
            
        Returns
        -------
        dedupe.Dedupe
            A trained dedupe model instance.
    """
    # To train dedupe, we feed it a sample of records.
    sample_num = math.floor(len(data) * sample_size)
    deduper.prepare_training(data, sample_size=sample_num)

    print('Starting active labeling...')

    dedupe.console_label(deduper)

    # Using the examples we just labeled, train the deduper and learn
    # blocking predicates
    deduper.train()

    # When finished, save our training to disk
    with open(training_file, 'w') as tf:
        deduper.write_training(tf)

    # Save our weights and predicates to disk.
    with open(settings_file, 'wb') as sf:
        deduper.write_settings(sf)
    
    return deduper

def _train(settings_file, training_file, data, field_properties, sample_size, update_model, n_cores):
    """Internal method that trains the deduper model from scratch or update
        an existing dedupe model.
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
        update_model : bool, default False
            If True, it allows user to update existing model by uploading
            training file.
        n_cores : int, default None
            Specify the number of cores to use during clustering.
            By default n_cores is equal to None (i.e. use multipressing equal to CPU count).
        Returns
        -------
        dedupe.Dedupe
            A dedupe model instance.
    """
    # Define the fields dedupe will pay attention to
    fields = []
    select_fields(fields, field_properties)
    
    if update_model == False:
        
        # If a settings file already exists, we'll just load that and skip training
        if os.path.exists(settings_file):
            print('Reading from', settings_file)
            with open(settings_file, 'rb') as f:
                deduper = dedupe.StaticDedupe(f, num_cores=n_cores)
        
        #Create a new deduper object and pass our data model to it.
        else:
            # Initialise dedupe
            deduper = dedupe.Dedupe(fields, num_cores=n_cores)
            
            # Launch active learning
            deduper = _active_learning(data, sample_size, deduper, training_file, settings_file)
            
    else:
        # ## Training
        # Initialise dedupe
        deduper = dedupe.Dedupe(fields, num_cores=n_cores)
        
        # Import existing model
        print('Reading labeled examples from ', training_file)
        with open(training_file, 'rb') as f:
            deduper.prepare_training(data, training_file=f)
        
        # Launch active learning
        deduper = _active_learning(data, sample_size, deduper, training_file, settings_file)

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
    print('Clustering...')
    clustered_dupes = deduper.partition(data, threshold)

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
                     config_name="dedupe_dataframe", update_model=False, threshold=0.4,
                     sample_size=0.3, n_cores=None):
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
        update_model : bool, default False
            If True, it allows user to update existing model by uploading
            training file. 
        threshold : float, default 0.4
           Only put together records into clusters if the cophenetic similarity of the cluster 
           is greater than the threshold.
        sample_size : float, default 0.3
            Specify the sample size used for training as a float from 0 to 1.
            By default it is 30% (0.3) of our data.
        n_cores : int, default None
            Specify the number of cores to use during clustering.
            By default n_cores is equal to None (i.e. use multipressing equal to CPU count).
    
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

    print('Importing data ...')

    df = clean_punctuation(df)
    
    specify_type(df, field_properties)                
    
    df['dictionary'] = df.apply(
        lambda x: dict(zip(df.columns, x.tolist())), axis=1)
    data_d = dict(zip(df.index, df.dictionary))

    # Train or load the model
    deduper = _train(settings_file, training_file, data_d, field_properties,
                     sample_size, update_model, n_cores)

    # Cluster the records
    clustered_df = _cluster(deduper, data_d, threshold, canonicalize)
    results = df.join(clustered_df, how='left')
    results.drop(['dictionary'], axis=1, inplace=True)

    return results
