```python
def cat_seq_features(group, col, name=None, nunique=True):
    name = col if name is None else name
    dummies = pd.get_dummies(group[col])
    mask = dummies.mask(dummies != 1)
    counts = mask.cumsum().sum(axis=1)
    df = counts.to_frame(name="{}_count".format(name))
    df["{}_ratio".format(name)] = counts / group.shape[0]
    if nunique:
        df["{}_nunique".format(name)] = mask.ffill().sum(axis=1)
    return df
```