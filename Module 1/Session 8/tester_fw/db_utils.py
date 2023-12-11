### Global Imports
# Some functionality needed by the notebook and demo cells:
from pprint import pprint, pformat
import math

# === Messages === #

def status_msg(s, verbose=True, **kwargs):
    if verbose:
        print(s, **kwargs)

# === Input/output === #

# def load_df_from_file(basename, dirname='resource/asnlib/publicdata/', abort_on_error=False, verbose=False):
def load_df_from_file(basename, dirname='', abort_on_error=False, verbose=False):
    from os.path import isfile
    from dill import loads
    from pandas import DataFrame
    df = DataFrame()
    filename = f"{dirname}{basename}"
    status_msg(f"Loading `DataFrame` from '{filename}'...", verbose=verbose)
    if isfile(filename):
        try:
            with open(filename, "rb") as fp:
                df = loads(fp.read())
            status_msg(f"  ==> Done!", verbose=verbose)
        except:
            if abort_on_error:
                raise
            else:
                df = DataFrame()
                status_msg(f"  ==> An error occurred.", verbose=verbose)
    return df

# def load_obj_from_file(basename, dirname='resource/asnlib/publicdata/', abort_on_error=False, verbose=False):
def load_obj_from_file(basename, dirname='', abort_on_error=False, verbose=False):
    from os.path import isfile
    from dill import loads
    from pandas import DataFrame
    filename = f"{dirname}{basename}"
    status_msg(f"Loading object from '{filename}'...", verbose=verbose)
    if isfile(filename):
        try:
            with open(filename, "rb") as fp:
                df = loads(fp.read())
            status_msg(f"  ==> Done! Type: `{type(df)}`", verbose=verbose)
        except:
            if abort_on_error:
                raise
            else:
                df = DataFrame()
                status_msg(f"  ==> An error occurred.", verbose=verbose)
    else:
        df = None
    return df

# def load_table_from_db(table_name, basename, dirname="resource/asnlib/publicdata/", verbose=False):
def load_table_from_db(table_name, basename, dirname="", verbose=False):
    from sqlite3 import connect
    from pandas import read_sql
    filename = f"{dirname}{basename}"
    if verbose:
        print(f"Retrieving table `{table_name}` from SQLite3 DB `{filename}`...")
    conn = connect(f"file:{filename}?mode=ro", uri=True)
    df = read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if verbose:
        print(f"... done! Found {len(df)} rows.")
    return df