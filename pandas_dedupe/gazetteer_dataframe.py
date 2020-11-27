from pandas_dedupe.utility_functions import (
    clean_punctuation,
    select_fields,
    specify_type
)

import os
import io
import logging
import math

import dedupe
import pandas as pd


logging.getLogger().setLevel(logging.WARNING)


def _active_learning(clean_data, messy_data, sample_size, deduper, training_file, settings_file):
    """Internal method that trains the deduper model using active learning.
        Parameters
        ----------
        clean_data : dict
            The dictionary form of the gazette that gazetteer_dedupe requires.
        messy_data : dict
            The dictionary form of the messy data that needs to be deduplicated 
            (and canonicalized)
        sample_size : float, default 0.3
            Specify the sample size used for training as a float from 0 to 1.
            By default it is 30% (0.3) of our data.
        deduper : a gazetteer model instance
        training_file : str
            A path to a training file that will be loaded to keep training
            from.
        settings_file : str
            A path to a settings file that will be loaded if it exists.
            
        Returns
        -------
        dedupe.Gazetteer
            A trained gazetteer model instance.
    """
    # To train dedupe, we feed it a sample of records.
    sample_num = math.floor(len(messy_data) * sample_size)
    deduper.prepare_training(clean_data, messy_data, sample_size=sample_num)

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

def _train(settings_file, training_file, clean_data, messy_data, field_properties, sample_size, update_model, n_cores):
    """Internal method that trains the deduper model from scratch or update
        an existing dedupe model.
        Parameters
        ----------
        settings_file : str
            A path to a settings file that will be loaded if it exists.
        training_file : str
            A path to a training file that will be loaded to keep training
            from.
        clean_data : dict
            The dictionary form of the gazette that gazetteer_dedupe requires.
        messy_data : dict
            The dictionary form of the messy data that needs to be deduplicated 
            (and canonicalized)
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
        dedupe.Gazetteer
            A gazetteer model instance.
    """
    # Define the fields dedupe will pay attention to
    fields = []
    select_fields(fields, [field_properties])
    
    if update_model == False:
        
        # If a settings file already exists, we'll just load that and skip training
        if os.path.exists(settings_file):
            print('Reading from', settings_file)
            with open(settings_file, 'rb') as f:
                deduper = dedupe.StaticGazetteer(f, num_cores=n_cores)
        
        #Create a new deduper object and pass our data model to it.
        else:
            # Initialise dedupe
            deduper = dedupe.Gazetteer(fields, num_cores=n_cores)
            
            # Launch active learning
            deduper = _active_learning(clean_data, messy_data, sample_size, deduper, training_file, settings_file)
            
    else:
        # ## Training
        # Initialise dedupe
        deduper = dedupe.Gazetteer(fields, num_cores=n_cores)

        # Import existing model
        print('Reading labeled examples from ', training_file)
        with open(training_file, 'rb') as f:
            deduper.prepare_training(clean_data, messy_data, training_file=f)

        # Launch active learning
        deduper = _active_learning(clean_data, messy_data, sample_size, deduper, training_file, settings_file)

    return deduper


def _cluster(deduper, clean_data, messy_data, threshold, canonicalize):
    """Internal method that clusters the data.
        Parameters
        ----------
        deduper : dedupe.Gazetteer
            A trained instance of gazetteer dedupe.
        clean_data : dict
            The dictionary form of the gazette that gazetteer_dedupe requires.
        messy_data : dict
            The dictionary form of the messy data that needs to be deduplicated 
            (and canonicalized)
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
    deduper.index(clean_data)                       
    
    clustered_dupes = deduper.search(messy_data, threshold, n_matches=None, generator=False)
    print('# duplicate sets', len(clustered_dupes))

    # Convert data_d to string so that Price & LatLong won't get traceback
    # during dedupe.canonicalize()
    for i in messy_data.values():
        for key in i:
            if i[key] is None:
                pass
            else:
                i[key] = str(i[key])
    
    df_data = []
    # ## Writing Results    
    for _, (messy_id, matches) in enumerate(clustered_dupes):
        for canon_id, scores in matches:
            
            tmp = {
                'cluster id': canon_id,
                'confidence': scores, 
                'record id': messy_id
            }
            df_data.append(tmp)
    
    # Add canonical name
    if canonicalize:
        clean_data_dict = pd.DataFrame.from_dict(clean_data).T.add_prefix('canonical_')
        clustered_df = (pd.DataFrame.from_dict(df_data)        # Create cluster result dataframe
                        .set_index('cluster id', drop=False)   # Note: cluster id is the index of clean_data (i.e. gazette)
                        .join(clean_data_dict, how='left')     # join clustered results and gazette
                        .set_index('record id')                # Note: record id is the index of the messy_data
                       )
    else:
        clustered_df = (pd.DataFrame.from_dict(df_data)        # Create clustered results dataframe
                        .set_index('record id')                # Note: record id is the index of messy_data
                       )
                        
    # Drop duplicates (i.e. keep canonical name with max confidence)
    # Note: the reason for this is that gazetteer dedupe might assign the same obs to multiple clusters
    confidence_maxes = clustered_df.groupby([clustered_df.index])['confidence'].transform(max) # Calculate max confidence
    clustered_df = clustered_df.loc[clustered_df['confidence'] == confidence_maxes]   # Keep rows with max confidence 
    clustered_df = clustered_df.loc[~clustered_df.index.duplicated(keep='first')]     # If same confidence keep the first obs
                   
    return clustered_df


def gazetteer_dataframe(clean_data, messy_data, field_properties, canonicalize=False,
                     config_name="gazetteer_dataframe", update_model=False, threshold=0.3,
                     sample_size=1, n_cores=None):
    """Deduplicates a dataframe given fields of interest.
        Parameters
        ----------
        clean_data : pd.DataFrame
            The gazetteer dataframe.
        messy_data : pd.DataFrame
            The dataframe to deduplicate.
        field_properties : str
            A string specifying what fields to use for deduplicating records.
        canonicalize : bool or list, default False
            Option that provides the canonical records as additional columns.
            Specifying a list of column names only canonicalizes those columns.
        setting_file : str, default None.
            the default name of the setting file is dedupe_dataframe_settings if None is provided.
        training_file : str, default None
            the default name of the setting file is dedupe_dataframe_training.json if None is provided.
            Note: the name of the training file should include the .json extension.
        update_model : bool, default False
            If True, it allows user to update existing model by uploading
            training file. 
        threshold : float, default 0.3
           only consider put together records into clusters if the cophenetic similarity of the cluster 
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
    assert type(clean_data)==pd.core.frame.DataFrame, 'Please provide a gazette in pandas dataframe format'
    assert len(clean_data.columns)==1, 'Please provide a gazetteer dataframe made of a single variable'
    assert type(field_properties) == str, 'field_properties must be in string (str) format'

    # Common column name
    common_name = clean_data.columns[0]
    
    # Canonical dataset (i.e. gazette)
    df_canonical = clean_punctuation(clean_data)
    df_canonical.rename(columns={field_properties: common_name}, inplace=True)
    specify_type(df_canonical, [common_name])                
    
    df_canonical['dictionary'] = df_canonical.apply(
        lambda x: dict(zip(df_canonical.columns, x.tolist())), axis=1)
    canonical = dict(zip(df_canonical.index, df_canonical.dictionary))
    
    # Messy dataset
    df_messy = clean_punctuation(messy_data)
    df_messy.rename(columns={field_properties: common_name}, inplace=True)
    specify_type(df_messy, [common_name])                

    df_messy['dictionary'] = df_messy.apply(
        lambda x: dict(zip(df_messy.columns, x.tolist())), axis=1)
    messy = dict(zip(df_messy.index, df_messy.dictionary))
    
    # Train or load the model
    deduper = _train(settings_file, training_file, canonical, messy, common_name,
                     sample_size, update_model, n_cores)
    
    # Cluster the records
    clustered_df = _cluster(deduper, canonical, messy, threshold, canonicalize)
    results = messy_data.join(clustered_df, how='left')
    results.rename(columns={'canonical_'+str(common_name): 'canonical_'+str(field_properties)}, inplace=True)

    return results
