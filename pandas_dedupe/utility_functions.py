from unidecode import unidecode
import pandas as pd

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
