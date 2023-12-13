# m1s8nb1 Data Debugging Part 1 Solutions:

## Exercise 1 Corrected Function:

Notes:
- Need to format the string so the parameters are interpolated properly.
- Need to include the GROUP BY clause.
- Need to add an alias so that COUNT(*) is named `count`.

```python
def count_interactions_by(col, conn):
    query = f"SELECT {col}, COUNT(*) AS count FROM Interactions GROUP BY {col}"
    return pd.read_sql(query, conn)
```

------------------------------------------------------

## Exercise 5 Corrected Function

Notes:
- We need to include records which are greater than
  or equal to our threshold, not just greater than
  the threshold.
  `uudf = uudf[uudf['count'] > threshold]` needs to be
  changed.
- We need a line which removes user similarities with
  themselves.
- Our reasoning about the symmetric dataframe is
  incorrect. The following lines should be removed:
```python
  symmetric_uudf = uudf.rename(columns={
      "user_id_x": "user_id_y",
      "user_id_y": "user_id_x"}
  )
  uudf = pd.concat((uudf, symmetric_uudf))
```

Here is the correct solution:

```python
### Exercise 5 solution
def connect_users(ubdf, threshold):
    ### BEGIN SOLUTION
    uudf = ubdf.merge(ubdf, on='book_id') \
               .groupby(['user_id_x', 'user_id_y']) \
               .size() \
               .reset_index() \
               .rename(columns={0: 'count'})
    uudf = uudf[uudf['user_id_x'] != uudf['user_id_y']]
    uudf = uudf[uudf['count'] >= threshold]
    uudf = uudf.reset_index(drop=True)
    return uudf
    ### END SOLUTION
```

------------------------------------------------------


## Exercise 6 Corrected Function:

Notes:
- We need to make `all_cids` the length of the `uids` variable,
  not the `communities` variable. Change `len(communities)` to
  len(`uids`). 
- The comm_id variable is a float instead of an integer.
  It needs to be converted to an int type.
  Better practice is to use `enumerate` instead of manually
  incrementing a floating point number.
- The comm_id variable needs to start at 0 instead of 1.
- The variables have "ID" in uppercase. They need to be
  lowercase.

```python
def assign_communities(communities):
    from pandas import DataFrame
    all_uids = []
    all_cids = []
    for cid, uids in enumerate(communities):
        all_uids += list(uids)
        all_cids += [cid] * len(uids)
    return DataFrame({'user_id': all_uids, 'comm_id': all_cids})
```

------------------------------------------------------


## Exercise 7 Corrected Function: 

Notes:
- We need to index by only the columns we're interested in.
  Do this by specifying our desired values and appending
  `df.groupby('comm_id')[VALUES]` to the third line of the
  function body.
- We need to reset our index so that it matches the
  structure of the desired output. Add the `.reset_index()`
  method to the end of the third line.
- We're currently performing an outer join instead of an
  inner join. Remove the `how='outer'` parameter from the
  first line.

```python
def means_by_community(intdf, comdf):
    ### BEGIN SOLUTION
    VALUES = ['is_read', 'rating', 'is_reviewed']
    df = intdf.merge(comdf, on='user_id')
    df = df.groupby('comm_id')[VALUES].mean().reset_index()
    return df
```

