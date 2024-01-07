def remove_tzs(s):
    return s.replace(' EDT', '').replace(' EST', '')

def load_dataset(dirname='resource/asnlib/publicdata/', **kwargs):
    from glob import glob
    from re import match
    from cse6040.utils import load_csv
    from pandas import to_datetime
    gfns = glob(f"{dirname}anon_grades_exam?.csv")
    gbe = {}
    for fn in gfns:
        if (m := match(r'^.*anon_grades_(exam\d+).csv$', fn)) is not None:
            exid = m.group(1)
            assert exid not in gbe
            gbe[exid] = load_csv(fn, dirname='', **kwargs).drop(columns=['Score'])

    tfns = glob(f"{dirname}anon_times_exam?.csv")
    tbe = {}
    for fn in tfns:
        if (m := match(r'^.*anon_times_(exam\d+).csv$', fn)) is not None:
            exid = m.group(1)
            assert exid not in tbe
            tbe[exid] = load_csv(fn, dirname='', **kwargs)
            # @FIXME: This hack is not correct but simplifies time processing
            for time_col in ['start date-time', 'submit date-time']:
                tbe[exid][time_col] = to_datetime(tbe[exid][time_col].apply(remove_tzs))
    ex_common = set(gbe.keys()) & set(tbe.keys())
    gbe = {exid: df for exid, df in gbe.items() if exid in ex_common}
    tbe = {exid: df for exid, df in tbe.items() if exid in ex_common}
    return gbe, tbe

# eof