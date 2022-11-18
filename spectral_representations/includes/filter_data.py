import numpy as np
import pandas as pd


def get_max(data, groupby="filename", col="val_accuracy"):
    new_data = []
    for filename, d in data.groupby(groupby):
        new_data.append(d.iloc[np.argmax(d[col])])
    return pd.DataFrame(new_data)


def get_corruption_to_level(data: pd.DataFrame, types=["brightness"]):
    new_data = []
    for i, row in data.iterrows():
        row = row.to_dict()
        for type in types:
            for t in [col for col in data.columns if col.startswith(type)]:
                i = float(t.split("_")[-1])
                new_data.append({**row, **dict(strength=i, corrupt=type, value=row[t])})
    return pd.DataFrame(new_data)