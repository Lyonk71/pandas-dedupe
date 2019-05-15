# ### dedupe left merge

# import modules
import pandas as pd
import pandas_dedupe

def left_join(dfa, dfb, field_properties):
    
    # create dataset label
    
    dfa["dataset"] = "dfa"
    dfb["dataset"] = "dfb"
    
    # concat datasets
    df_concat = pd.concat([dfb, dfa], sort=False)
    
    # cluster
    df_concat = pandas_dedupe.dedupe_dataframe(
        df_concat, field_properties 
    )
    
    # break apart
    dfa = df_concat[df_concat["dataset"] == "dfa"]
    dfb = df_concat[df_concat["dataset"] == "dfb"]
    
    # take min nulls per cluster for dataset b
    dfb = dfb.dropna(subset=["cluster id"])
    
    dfb["count"] = pd.isnull(dfb).sum(axis=1)
    dfb = dfb.sort_values(["count"])
    dfb = dfb.drop_duplicates(subset=["cluster id"], keep="first")
    dfb = dfb.drop(columns=["count"])
    
    # left merge b onto a
    dfa = dfa.dropna(axis=1, how="all")
    dfb = dfb.dropna(axis=1, how="all")
    
    dfb = dfb.rename({"cluster id": "clusterid", "confidence": "confidenc"}, axis=1)
    dfb = dfb[dfb.columns.difference(dfa.columns)]
    dfb = dfb.rename({"clusterid": "cluster id", "confidenc": "confidence"}, axis=1)
    
    dfa = dfa.merge(dfb, how="left", on="cluster id")
    
    return dfa
    #
