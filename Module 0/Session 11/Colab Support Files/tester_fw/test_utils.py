def canonicalize_tibble(X, remove_index=True):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=remove_index)
    Y.reset_index(drop=remove_index, inplace=remove_index)
    return Y

def assert_tibbles_left_matches_right(A, B, exact=False, sort_df=True, col_type=True):
    from pandas.testing import assert_frame_equal
    A_canonical = canonicalize_tibble(A, sort_df)
    B_canonical = canonicalize_tibble(B, sort_df)
    assert_frame_equal(A_canonical, B_canonical, check_exact=exact, check_dtype=col_type)

def assert_tibbles_are_equivalent(A, B, **kwargs):
    assert_tibbles_left_matches_right(A, B, **kwargs)

def compare_copies(a, b, tol=0, exact=False, sort_df=True, col_type=True):
    import pandas as pd
    import numpy as np
    try:
        # list or tuple
        if (isinstance(a, list) and isinstance(b, list))\
            or (isinstance(a, tuple) and isinstance(b, tuple)):
            if len(a) != len(b): return False
            return all(compare_copies(ai, bi) for ai, bi in zip(a, b))
        # set
        if isinstance(a, set) and isinstance(b, set):
            if len(a) != len(b): return False
            return not (a - b)
        # dict
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()): return False
            return all(compare_copies(va, b[ka]) for ka, va in a.items())
        # DataFrame
        if isinstance(a, pd.DataFrame):
            try:
                assert_tibbles_left_matches_right(a, b, exact, sort_df, col_type)
                return True
            except AssertionError:
                return False
        # Series
        if isinstance(a, (int, np.int64, float,  pd.Series)) \
            and isinstance(b, (int, np.int64, float,  pd.Series)):
            try:
                return np.isclose(a, b, atol=tol, rtol=0, equal_nan=True)
            except:
                pass
            try:
                return (a == b).all()
            except:
                raise
        # string or boolean
        elif isinstance(a, (str, bool,)) \
            and isinstance(b, (str, bool,)):

            return a == b
        
        elif isinstance(a, (np.ndarray,)) \
            and isinstance(b, (np.ndarray,)):

            return np.allclose(a, b, atol=tol, rtol=0, equal_nan=True)
        elif (a is None and b is None):
            return True
    except Exception as e:
        print(e.__traceback__)
        return False


    



def df_helper(returned_outputs, true_outputs):
    true_df = true_outputs['df']
    returned_df = returned_outputs['df']
    cols = true_df.columns.to_list()
    ret_cols = returned_df = returned_outputs['df'].columns.to_list()

    assert ret_cols == cols, 'Your result has different columns than the expected result. The columns must have the same names and be in the same order to pass this check. You can use the variables `returned_output_vars["df"]` and `true_output_vars["df"]` in your notebook to debug.'

    returned_df = returned_outputs['df'].sort_values(by=cols).reset_index(drop=True)
    true_df = true_df.sort_values(by=cols).reset_index(drop=True)

    assert true_df.equals(returned_df), 'Your result did not match the expected result. This check verifies that the `shape`, `dtypes` and `columns` attributes and the values of `returned_output_vars["df"]` and `true_output_vars["df"]` match. The rows can be sorted in any order. You can use the variables `returned_output_vars["df"]` and `true_output_vars["df"]` in your notebook to debug.'



def dfs_to_conn(conn_dfs, index=False):
    import sqlite3
    conn = sqlite3.connect(':memory:')
    for table_name, df in conn_dfs.items():
        # df.rename_axis(index='index').to_sql(table_name, conn, if_exists='replace')
        df.to_sql(table_name, conn, if_exists='replace', index=index)
    return conn

def get_memory_usage():
    import os
    import psutil
    print(psutil.Process(os.getpid()).rss//1024*2, 'mb')
