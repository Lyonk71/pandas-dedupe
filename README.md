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
    df_final.to_csv('test_output.csv')


* *A training file and a settings file will be created while running Dedupe. Keeping these files will eliminate the need to retrain your model in the future. If you would like to retrain your model, just delete the settings and training files.*

# Credits
