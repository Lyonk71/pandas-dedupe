# pandas-dedupe

The Dedupe library made easy with Pandas.

# Installation

```
pip install pandas-dedupe
```

# Video Tutorials

[Basic Deduplication](https://www.youtube.com/watch?v=lCFEzRaqoJA)

# Basic Usage

A training file and a settings file will be created while running Dedupe.
Keeping these files will eliminate the need to retrain your model in the future.

If you would like to retrain your model from scratch, just delete the settings and training files.

### Deduplication (dedupe_dataframe)
`dedupe_dataframe` is for deduplication when you have data that can contain multiple records that can all refer to the same entity

```python
import pandas as pd
import pandas_dedupe

#load dataframe
df = pd.read_csv('test_names.csv')

#initiate deduplication
df_final = pandas_dedupe.dedupe_dataframe(df,['first_name', 'last_name', 'middle_initial'])

#send output to csv
df_final.to_csv('deduplication_output.csv')
```

### Gazetteer deduplication (gazetteer_dataframe)
`gazetteer_dataframe` is for matching a messy dataset against a 'canonical dataset' (i.e. the gazette)

```python
import pandas as pd
import pandas_dedupe

#load dataframe
df_clean = pd.read_csv('gazette.csv')
df_messy = pd.read_csv('test_names.csv')

#initiate deduplication
df_final = pandas_dedupe.gazetteer_dataframe(df_clean, df_messy, 'fullname', canonicalize=True)

#send output to csv
df_final.to_csv('gazetteer_deduplication_output.csv')
```


### Matching / Record Linkage

Use identical field names when linking dataframes.
Record linkage should only be used on dataframes that have been deduplicated.

```python
import pandas as pd
import pandas_dedupe

#load dataframes
dfa = pd.read_csv('file_a.csv')
dfb = pd.read_csv('file_b.csv')

#initiate matching
df_final = pandas_dedupe.link_dataframes(dfa, dfb, ['field_1', 'field_2', 'field_3', 'field_4'])

#send output to csv
df_final.to_csv('linkage_output.csv')
```

# Advanced Usage

### Canonicalize Fields

The canonicalize parameter will standardize names in a given cluster. Original fields are also kept.

```python
pandas_dedupe.dedupe_dataframe(df,['first_name', 'last_name', 'payment_type'], canonicalize=True)
```

### Update Threshold (dedupe_dataframe and gazetteer_dataframe only)

Group records into clusters only if the cophenetic similarity of the cluster is greater than
the threshold.

```python
pandas_dedupe.dedupe_dataframe(df, ['first_name', 'last_name'], threshold=.7)
```

### Update Existing Model (dedupe_dataframe and gazetteer_dataframe only)

If `True`, it allows a user to update the existing model.

```python
pandas_dedupe.dedupe_dataframe(df, ['first_name', 'last_name'], update_model=True)
```

### Recall Weight & Sample Size

The `dedupe_dataframe()` function has two optional parameters specifying `recall_weight` and `sample_size`:

- **recall_weight** - Ranges from 0 to 2. When set to 2, we are saying we care twice as much
  about recall than we do about precision.
- **sample_size** - Specifies the sample size used for training as a float from 0 to 1.
  By default it is 30% (0.3) of our data.

### Specifying Types

If you'd like to specify dates, spatial data, etc, do so here. The structure must be like so:
`('field', 'type', 'additional_parameter)`. the `additional_parameter` section can be omitted.
The default type is `String`.

See the full list of types [below](#Types).

```python
# Price Example
pandas_dedupe.dedupe_dataframe(df,['first_name', 'last_name', ('salary', 'Price')])

# has missing Example
pandas_dedupe.link_dataframes(df,['SSN', ('bio_pgraph', 'Text'), ('salary', 'Price', 'has missing')])

# crf Example
pandas_dedupe.dedupe_dataframe(df,[('first_name', 'String', 'crf'), 'last_name', (m_initial, 'Exact')])
```

# Types

Dedupe supports a variety of datatypes; a full list with documentation can be found [here.](https://docs.dedupe.io/en/latest/Variable-definition.html#)

pandas-dedupe officially supports the following datatypes:

- **String** - Standard string comparison using string distance metric. This is the default type.
- **Text** - Comparison for sentences or paragraphs of text. Uses cosine similarity metric.
- **Price** - For comparing positive, non zero numerical values.
- **DateTime** - For comparing dates.
- **LatLong** - (39.990334, 70.012) will not match to (40.01, 69.98) using a string distance
  metric, even though the points are in a geographically similar location. The LatLong type resolves
  this by calculating the haversine distance between compared coordinates. LatLong requires
  the field to be in the format (Lat, Long). The value can be a string, a tuple containing two
  strings, a tuple containing two floats, or a tuple containing two integers. If the format
  is not able to be processed, you will get a traceback.
- **Exact** - Tests whether fields are an exact match.
- **Exists** - Sometimes, the presence or absence of data can be useful in predicting a match.
  The Exists type tests for whether both, one, or neither of fields are null.

Additional supported parameters are:

- **has missing** - Can be used if one of your data fields contains null values
- **crf** - Use conditional random fields for comparisons rather than distance metric. May be more
  accurate in some cases, but runs much slower. Works with String and ShortString types.

# Contributors

[Tyler Marrs](http://tylermarrs.com/) - Refactored code, added docstrings, added `threshold` parameter

[Tawni Marrs](https://github.com/tawnimarrs) - refactored code, added docstrings

[ieriii](https://github.com/ieriii) - Added `update_model` parameter, updated codebase to use `Dedupe 2.0`, added support for multiprocessing, added `gazetteer_dataframe`.

[Daniel Marczin](https://github.com/dim5) - Extensive updates to documentation to enhance readability.

[Alexis-Evelyn](https://github.com/alexis-evelyn) - Fixed logger warning with related to Pandas.

[Niels Horn](https://github.com/nilq) - Cleaned up utility functions.

# Credits

Many thanks to folks at [DataMade](https://datamade.us/) for making the the [Dedupe library](https://github.com/dedupeio/dedupe) publicly available. People interested in a code-free implementation of the dedupe library can find a link here: [Dedupe.io](https://dedupe.io/pricing/).
