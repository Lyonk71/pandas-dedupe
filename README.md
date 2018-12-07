# pandas-dedupe
The Dedupe library made easy with Pandas.

# Installation

# Usage

### Basic Deduplication

    import pandas as pd
    import pandas_dedupe

    #load dataframe
    df = pd.read_csv('test_names.csv')

    #initiate deduplication
    df_final = pandas_dedupe.dedupe_dataframe(df,['first_name', 'last_name', 'middle_initial'])

    #send output to csv
    df_final.to_csv('deduplication_output.csv')
    
    
    #------------------------------additional details------------------------------

    #A training file and a settings file will be created while running Dedupe. 
    #Keeping these files will eliminate the need to retrain your model in the future. 
    #If you would like to retrain your model, just delete the settings and training files.

### Basic Matching / Record Linkage

    import pandas as pd
    import pandas_dedupe

    #load dataframes
    dfa = pd.read_csv('file_a.csv')
    dfb = pd.read_csv('file_b.csv')
    
    #initiate matching
    df_final = pandas_dedupe.link_dataframes(dfa, dfb, ['field_1', 'field_2', 'field_3', 'field_4'])

    #send output to csv
    df_final.to_csv('linkage_output.csv')
    
    
    #------------------------------additional details------------------------------
    
    #Use identical field names when linking dataframes.
    
    #Record linkage should only be used on dataframes that have been deduplicated.
       
    #A training file and a settings file will be created while running Dedupe. 
    #Keeping these files will eliminate the need to retrain your model in the future. 
    #If you would like to retrain your model, just delete the settings and training files.


# Credits

Many thanks to folks at [DataMade](https://datamade.us/) for making the the [Dedupe library](https://github.com/dedupeio/dedupe) publicly available. People interested in a code-free implementation of the dedupe library can find a link here: [Dedupe.io](https://dedupe.io/pricing/).

