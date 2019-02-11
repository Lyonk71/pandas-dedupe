# pandas-dedupe
The Dedupe library made easy with Pandas.

# Installation

pip install pandas-dedupe

# Video Tutorials

[Basic Deduplication](https://www.youtube.com/watch?v=lCFEzRaqoJA)

# Basic Usage

### Deduplication

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

### Matching / Record Linkage

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
    
# Advanced Usage

### Set Confidence Threshold

    pandas_dedupe.dedupe_dataframe(df,['first_name', 'last_name', 'middle_initial'], confidence_threshold=.3)
    
    #------------------------------additional details------------------------------
    
    #The default confidence_threshold is .2.
    
    #The maximum value needed by at least one score in a cluster for the cluster to
    #be recorded. For instance, with a cluster threshold of .3, the cluster below 
    #would be kept:
    
    #| name      | confidence | cluster id |
    #|-----------|------------|------------|
    #| John      | .7         | 0          |
    #| Johnny    | .6         | 0          |
    #| Johnathon | .4         | 0          |
    
    #In the same case, the following cluster would not be recognized:
    
    #| name      | confidence | cluster id |
    #|-----------|------------|------------|
    #| Ringo     | .2         |            |
    #| Ronald    | .2         |            |

### Specifying Types

    # Price Example
    pandas_dedupe.dedupe_dataframe(df,['first_name', 'last_name', ('salary', 'Price')])
       
    # has missing Example
    pandas_dedupe.link_dataframes(df,['SSN', ('bio_pgraph', 'Text'), ('salary', 'Price', 'has missing')])
    
    # crf Example
    pandas_dedupe.dedupe_dataframe(df,[('first_name', 'String', 'crf'), 'last_name', (m_initial, 'Exact')])
    
    
    #------------------------------additional details------------------------------
    
    #If a type is not explicity listed, String will be used.
    
    #Tuple (parenthesis) is required to declare all other types. If you prefer use tuple
    #for string also, ('first_name', 'String'), that's fine.
    
    #If you want to specify either a 'crf' or 'has missing' parameter, a tuple with three elements
    #must be used. ('first_name', 'String', 'crf') works, ('first_name', 'crf') does not work.
        
# Types

Dedupe supports a variety of datatypes; a full listing with documentation can be found [here.](https://docs.dedupe.io/en/latest/Variable-definition.html#)

pandas-dedupe officially supports the following datatypes:
* **String** - Standard string comparison using string distance metric. This is the default type.
* **Text** - Comparison for sentences or paragraphs of text. Uses cosine similarity metric.
* **Price** - For comparing positive, non zero numerical values.
* **DateTime** - For comparing dates.
* **LatLong** - (39.990334, 70.012) will not match to (40.01, 69.98) using a string distance
metric, even though the points are in a geographically similar location. The LatLong type resolves
 this by calculating the haversine distance between compared coordinates. LatLong requires
 the field to be in the format (Lat, Lng). The value can be a string, a tuple containing two
 strings, a tuple containing two floats, or a tuple containing two integers. If the format
 is not able to be processed, you will get a traceback.
* **Exact** - Tests wheter fields are an exact match.
* **Exists** - Sometimes, the presence or absence of data can be useful in predicting a match.
The Exists type tests for whether both, one, or neither of fields are null.

Additional supported parameters are:
* **has missing** - Can be used if one of your data fields contains null values
* **crf** - Use conditional random fields for comparisons rather than distance metric. May be more
accurate in some cases, but runs much slower. Works with String and ShortString types.

# Credits

Many thanks to folks at [DataMade](https://datamade.us/) for making the the [Dedupe library](https://github.com/dedupeio/dedupe) publicly available. People interested in a code-free implementation of the dedupe library can find a link here: [Dedupe.io](https://dedupe.io/pricing/).

