import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# === Iteration === #

def isiter(x):
    """
    Returns `True` if `x` is iterable.

    Uses the "duck typing" method described here:
    https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    """
    try:
        iterator = iter(x)
    except TypeError:
        return False
    return True

def sample_iter(I, n=1, rng_or_seed=None, replace=False, safe=True):
    from pandas import DataFrame, Series
    from numpy import ndarray, array
    
    rng = get_rng(rng_or_seed, ret_type=False)
    n_sample = min(n, len(I)) if safe else n
    sample_locs = rng.choice(range(len(I)), size=n_sample, replace=replace)
    if isinstance(I, DataFrame) or isinstance(I, Series):
        sample = I.iloc[sample_locs]
    elif isinstance(I, ndarray):
        sample = I[sample_locs]
    elif isinstance(I, list):
        sample = [I[k] for k in sample_locs]
    elif isinstance(I, dict):
        K = list(I.keys())
        sample_values = [K[k] for k in sample_locs]
        sample = {k: I[k] for k in sample_values}
    else:
        J = array(list(I))
        sample = type(I)(J[sample_locs])
    return sample

# === Messages === #

def status_msg(s, verbose=True, **kwargs):
    if verbose:
        print(s, **kwargs)

# === pandas ===

def subselect(df, col, values):
    """
    Subselects rows of a `DataFrame` where the column `col`
    contains any of the given `values`.

    If `values` is a non-iterable object _or_ a `str`,
    then this function treats it as a single value to
    find.
    """
    if not isinstance(values, str) and isiter(values):
        return df[df[col].isin(values)]
    return df[df[col] == values]

# === Input/output === #

def load_csv(basename, dirname='resource/asnlib/publicdata/',
             abort_on_error=True, verbose=True,
             **kwargs):
    from os.path import isfile
    from pandas import read_csv
    filename = f"{dirname}{basename}"
    status_msg(f"Loading `DataFrame` from '{filename}'...", verbose=verbose)
    if isfile(filename):
        try:
            df = read_csv(filename, **kwargs)
            status_msg(f"  ==> Done!", verbose=verbose)
        except:
            if abort_on_error:
                raise
            else:
                df = DataFrame()
                status_msg(f"  ==> An error occurred.", verbose=verbose)
    return df

def text_to_file(s, basename, dirname='resource/asnlib/publicdata/', overwrite=True, verbose=True):
    from os.path import isfile
    filename = f"{dirname}{basename}"
    status_msg(f"Writing string to '{filename}'...", verbose=verbose)
    if not overwrite and isfile(filename):
        status_msg(f"  ==> File exists already; skipping.", verbose=verbose)
    else:
        with open(filename, "wt") as fp:
            fp.write(s)
        status_msg(f"  ==> Done!", verbose=verbose)

def load_text_from_file(basename, dirname='resource/asnlib/publicdata/', abort_on_error=False, verbose=False):
    from os.path import isfile
    filename = f"{dirname}{basename}"
    status_msg(f"Loading string from '{filename}'...", verbose=verbose)
    if isfile(filename):
        try:
            with open(filename, "rt") as fp:
                s = fp.read()
            status_msg(f"  ==> Done!", verbose=verbose)
        except:
            if abort_on_error:
                raise
            else:
                status_msg(f"  ==> An error occurred.", verbose=verbose)
                s = ''
    return s

def df_to_file(df, basename, dirname='resource/asnlib/publicdata/', overwrite=True, verbose=True):
    from os.path import isfile
    from dill import dumps
    filename = f"{dirname}{basename}"
    if verbose:
        print(f"Writing `DataFrame` to '{filename}'...")
    if not overwrite and isfile(filename):
        print(f"  ==> File exists already; skipping.")
    else:
        with open(filename, "wb") as fp:
            fp.write(dumps(df))
        print(f"  ==> Done!")

def load_df_from_file(basename, dirname='resource/asnlib/publicdata/', abort_on_error=False, verbose=False):
    from os.path import isfile
    from dill import loads
    from pandas import DataFrame
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

def obj_to_file(df, basename, dirname='resource/asnlib/publicdata/', overwrite=True, verbose=True):
    from os.path import isfile
    from dill import dumps
    filename = f"{dirname}{basename}"
    if verbose:
        print(f"Writing object (type `{type(df)}`) to '{filename}'...")
    if not overwrite and isfile(filename):
        print(f"  ==> File exists already; skipping.")
    else:
        with open(filename, "wb") as fp:
            fp.write(dumps(df))
        print(f"  ==> Done!")

def load_obj_from_file(basename, dirname='resource/asnlib/publicdata/', abort_on_error=False, verbose=False):
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

def load_table_from_db(table_name, basename, dirname="resource/asnlib/publicdata/", verbose=False):
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

# ==== RNGs ==== #

@static_vars(DEFAULT_RNG=None)
def get_rng(rng_or_seed, ret_type=True):
    """
    Returns a valid pseudorandom-number generator (RNG) object
    based on how `rng_or_seed` is set:

    - An integer: Creates a new RNG with the integer as a seed
    - An existing RNG object: Returns the same object
    - `None`: Returns a global "default" RNG, which is created
      once at module initialization time.

    If `ret_type` is set, this function also returns a descriptive
    string saying which of the above cases applies. (The intent of
    this string is for use in printing as part of debugging output.)
    """
    # Initialize static variable, DEFAULT_RNG
    from numpy.random import default_rng
    if get_rng.DEFAULT_RNG is None:
        get_rng.DEFAULT_RNG = default_rng(1_234_567_890)

    if isinstance(rng_or_seed, int):
        rng = default_rng(rng_or_seed)
        rng_type = f'`default_rng({rng_or_seed})`'
    elif rng_or_seed is None:
        rng = get_rng.DEFAULT_RNG
        rng_type = f'`DEFAULT_RNG` [{rng}]'
    else:
        rng = rng_or_seed # had better be a RNG
        rng_type = f'User-supplied [{rng}]'

    return (rng, rng_type) if ret_type else rng

# === Random word generator ===
@static_vars(words=None)
def random_words(dirname="resource/asnlib/publicdata/", verbose=False, **kwargs):
    if random_words.words is None:
        infile = f"{dirname}1-1000.txt"
        if verbose:
            print(f"Loading common English words from: '{infile}' ...")
        with open(infile, "rt") as fp:
            random_words.words = [line.strip() for line in fp.readlines()]
    assert random_words.words is not None
    return sample_iter(random_words.words, **kwargs)

# === Miscellaneous calculations === #
def make_logbins(x, base=np.e, verbose=False):
    from numpy import log, log2, log10, floor, ceil, arange, concatenate
    
    if base == 2:
        logger = log2
    elif base == 10:
        logger = log10
    elif base == np.e:
        logger = log
    else:
        logger = lambda x: log(x) / log(base)
    if verbose: print(f"[make_logbins({id(x)}, base={base})] logger={logger.__name__}")
    
    pmin = int(floor(logger(x.min())))
    pmax = int(ceil(logger(x.max()))) + 1
    if verbose: print(f"[make_logbins({id(x)}, base={base})] pmin={pmin}, pmax={pmax}")
    bins = float(base)**arange(pmin, pmax)
    if verbose: print(f"[make_logbins({id(x)}, base={base})] bins={bins}")
    return bins

def make_ecdf(x):
    x, f = np.unique(x, return_counts=True)
    F = f.cumsum() / f.sum()
    x = np.concatenate(([0.0], x))
    return F, x

def fit_sequenced_observations(b, verbose=True):
    A = np.array([np.arange(1, len(b)+1), np.ones(len(b))]).T
    model = sm.OLS(b, A)
    fit = model.fit()
    if verbose:
        print(fit.summary())
    return fit

def fmt_pow(x, base=None, max_len=5):
    from numpy import e, log, log2, log10
    if base == 2:
        logger = log2
    elif base == 10:
        logger = log10
    elif base == e:
        logger = log
    else:
        logger = lambda y: log(y) / log(base)
        
    if len(f"{x}") > max_len:
        return f"${base}^{{{logger(x):.0f}}}$"
    if x < 1:
        return f"{x}"
    return f"{x:,.0f}"

# ==== Plotting ==== #

def plot_lags(acorr, lags=None
              , xlabel=r'$h$ (lag)', ylabel=r'$\hat{\rho}(h)$'
              , title='Sample autocorrelation'
              , linestyle=':'
              , ax=None, figsize=(8, 8/16*9)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        
    if lags is None:
        lags = np.arange(len(acorr))
        
    ax.stem(lags, acorr, linefmt=linestyle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, horizontalalignment='right')
    ax.set_title(title)
    return ax

def plot_ecdf(obs
              , exp=True, exp_color='grey', exp_linestyle='dashed'
              , xlabel=r'$x$', ylabel=None, title='Empirical cumulative distribution function (ECDF)'
              , ax=None, figsize=(8, 8/16*9), ret_ecdf=False):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    F, x = make_ecdf(obs)
    ax.stairs(F, x, label=r'ECDF, $F(x)$')
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, horizontalalignment='right')
    ax.set_title(title)
    
    if exp:
        rate = 1 / obs.mean()
        exp_label = r'Exponential, $F_{\mathrm{exp}}(x) = 1 - e^{-\lambda x}$ where $\lambda=$' + f'{rate:.3g}'
        ax.plot(x, 1-np.exp(-rate*x), color=exp_color, linestyle=exp_linestyle, label=exp_label)
        
    ax.legend(loc='best')
    if ret_ecdf:
        return ax, (F, x)
    return ax

def plot_interarrival_times(times, color='black'
                            , mean=False, mean_color='grey'
                            , fit=None, fit_color='C1'
                            , title="Interarrival times", xlabel="Observation number", ylabel="Time (mins)"
                            , ax=None, figsize=(8, 8/16*9)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        
    x = np.arange(len(times))
    y = times
    ax.scatter(x, y, color=color)
    
    # Embellishment: show the mean value
    if mean:
        y_mean = y.mean()
        ax.axhline(y_mean, color=mean_color)
        ax.text(ax.get_xlim()[0], y_mean, f'mean', color=mean_color
                , horizontalalignment='left', verticalalignment='bottom')
        
    # Embellishment: show a fitted regression-line
    if fit is not None:
        x_fit = np.array(ax.get_xlim())
        y_fit = fit.params[0]*x_fit + fit.params[1]
        ax.plot(x_fit, y_fit, color=fit_color, linestyle='dashed')
        ax.text(x_fit[0], y_fit[0], f'best-fit line', color=fit_color
                , horizontalalignment='left', verticalalignment='bottom')
    
    # Text embellishments
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, horizontalalignment='right')
    return ax

def add_disjoint_windowed_means(ax, x, wins=[2, 3, 4, 8, 16]):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        
    for k, w in enumerate(wins):
        if w >= len(x): break
        y_djw, x_djw = disjoint_windowed_means(x, w)
        c_djw = f'C{k+1}' # color
        ax.plot(x_djw, y_djw, 'o:', color=c_djw)

        xt = x_djw[abs(y_djw).argmax()]
        yt = abs(y_djw).max()
        ax.text(xt, yt, f'w={w}', color=c_djw, horizontalalignment='left', verticalalignment='bottom')
        
    return ax

def plot_series_loglog(series, ax=None, figsize=(8, 8/16*9), **kwargs):
    from matplotlib.pyplot import figure, gca
    if ax is None:
        fig = figure(figsize=figsize)
        ax = gca()
    x = series.index
    y = series.values
    ax.loglog(x, y, '.', **kwargs)
    return ax

def display_image_from_file(filename, verbose=False):
    from IPython.display import Image
    if verbose:
        print(f"Loading image, `{filename}` ...")
    display(Image(filename))
    if verbose:
        print(f"... done! (Did it appear?)")

def scatter_hist(series, base=np.e, normalize=False, percentage=False
                 , ax=None, figsize=(8, 8/16*9)
                 , supertitle=None, title=None, xlabel=None, ylabel=None):
    from matplotlib.pyplot import figure, gca, suptitle
    from pandas import cut
    
    bins = make_logbins(series, base=base)
    counts = cut(series, bins=bins, labels=bins[:-1], right=False).value_counts()
    y = counts.copy()
    if normalize or percentage:
        y /= y.sum()
        if percentage:
            y *= 100
    
    if ax is None:
        fig = figure(figsize=figsize)
        ax = gca()
        
    ax.scatter(y.index, y.values)
    ax.set_xscale('log', base=base)
    ax.set_yscale('log', base=base)
    ax.set_aspect(1.0)
    suptitle(supertitle)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, horizontalalignment='right')
    
    ax.set_xticklabels([fmt_pow(v, base=base) for v in ax.get_xticks()]);
    ax.set_yticklabels([fmt_pow(v, base=base) for v in ax.get_yticks()]);

    return ax, bins, counts

def lineseg_hist(series, xbase=np.e, ybase=10, normalize=False, percentage=False
                 , ax=None, figsize=(8, 8/16*9)
                 , supertitle=None, title=None, xlabel=None, ylabel=None):
    from matplotlib.pyplot import figure, gca, suptitle
    from matplotlib.collections import LineCollection
    from pandas import cut

    bins = make_logbins(series, base=xbase)
    counts = cut(series, bins=bins, labels=bins[:-1], right=False).value_counts()
    y = counts.sort_index()
    if normalize or percentage:
        y /= y.sum()
        if percentage:
            y *= 100
    
    if ax is None:
        fig = figure(figsize=figsize)
        ax = gca()

    seg_starts = y.index
    seg_ends = list(y.index[1:]) + [bins[-1]]
    segs = [[(x0, y0), (x1, y0)] for x0, x1, y0 in zip(seg_starts, seg_ends, y.values)]
    segcol = LineCollection(segs, color='C0')
    ax.add_collection(segcol)
    
    ax.scatter(y.index, y.values)
    ax.set_xscale('log', base=xbase)
    ax.set_yscale('log', base=ybase)
    if xbase == ybase:
        ax.set_aspect(1.0)
    
    suptitle(supertitle)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, horizontalalignment='right')
    
    ax.set_xticks(bins)
    ax.set_xticklabels([fmt_pow(v, base=xbase) for v in ax.get_xticks()]);
    
    if normalize:
        ax.set_ylim(ax.get_ylim()[0], 1)
    elif percentage:
        ax.set_ylim(ax.get_ylim()[0], 100)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([fmt_pow(v, base=ybase) for v in ax.get_yticks()]);
    
    for x_, y_ in zip(y.index, y.values):
        ax.text(x_*(1.1), y_, f"{y_:.1f}%", verticalalignment='bottom')

    return ax, bins, counts

# ==== Graph / NetworkX interfacing ===== #

def to_nx(edge_list):
    from networkx import DiGraph
    G = DiGraph()
    G.add_weighted_edges_from(edge_list)
    return G

def graph_to_matrix(G):
    try:
        from networkx import to_scipy_sparse_array # Works in 3.0
        return to_scipy_sparse_array(G)
    except:
        pass

    try:
        from networkx import to_scipy_sparse_matrix # Works in 2.5
        return to_scipy_sparse_matrix(G)
    except:
        raise

def graph_spy(G, style='matrix', ax=None, figsize=(6.5, 6.5), **kwargs):
    from matplotlib.pyplot import figure, gca
    from networkx import spring_layout, draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels
    
    if ax is None:
        fig = figure(figsize=figsize)
        ax = gca()
        
    if style == 'matrix':
        A = graph_to_matrix(G)
        ax.spy(A, **kwargs)
    else:
        pos = spring_layout(G, seed=7)

        # nodes
        draw_networkx_nodes(G, pos) #, node_size=700)

        # edges
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.1]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.1]
        draw_networkx_edges(G, pos, edgelist=elarge, width=2)
        draw_networkx_edges(G, pos, edgelist=esmall, width=0.5, alpha=0.5)

        # node labels
        draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        # edge weight labels
#        edge_labels = nx.get_edge_attributes(G, "weight")
#        nx.draw_networkx_edge_labels(G, pos, edge_labels)
    return ax

def detect_communities(G, seed=1_234):
    from networkx.algorithms.community import louvain_communities
    return louvain_communities(G, seed=seed)

def random_clusters(nc, nvc, p_intra=0.5, p_inter=0.1, rng_or_seed=None, verbose=False):
    rng = get_rng(rng_or_seed, ret_type=False)
    n = nc * nvc
    mv_intra = int(p_intra*nvc) + 1 # no. of intra-cluster edges per vertex
    mv_inter = int(p_inter*n)       # no. of inter-cluster edges per vertex
    
    if verbose:
        print('Constructing a vertex-clustered graph with these properties:')
        print(f'- Number of clusters: nc={nc}')
        print(f'- Vertices per cluster: nvc={nvc}')
        print(f'- Number of intra-cluster edges per vertex: {mv_intra} (p_intra={p_intra})')
        print(f'- Number of inter-cluster edges per vertex: {mv_inter} (p_inter={p_inter})')
        print(f'- RNG: {rng}')
    
    V = set(range(n)) # `n` vertices
    E = []
    for c in range(nc):
        V_c = set(range(c*nvc, (c+1)*nvc))
        for v in V_c:
            # Add intra-cluster edges for `v`
            N_v = sample_iter(V_c - {v}, n=mv_intra, rng_or_seed=rng)
            W_v = rng.random(size=len(N_v))
            E += [(v, u, w) for u, w in zip(N_v, W_v)]
            
            # Add inter-cluster edges for `v`
            X_v = sample_iter(V - V_c, n=mv_inter, rng_or_seed=rng)
            W_v = rng.random(size=len(X_v)) / 10.0 # make these edges weaker, too
            E += [(v, u, w) for u, w in zip(X_v, W_v)]
    return E

# === bike trip stuff === #
def viz_availability(df, col_avail='net_avail'
                     , title="Net bikes available"
                     , ax=None, figsize=(8, 8/16*9)
                     , legend=False
                     , **kwargs):
    
    min_avail, max_avail = df[col_avail].min(), df[col_avail].max()
    
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
    df.plot(x='time', y=col_avail, style='.', ax=ax, **kwargs)
    if not legend:
        ax.get_legend().remove()
    
    ax.axhline(0, color='black', linestyle='solid', linewidth=.5)
    ax.axhline(min_avail, color='darkgrey', linestyle='dotted')
    ax.axhline(max_avail, color='darkgrey', linestyle='dotted')
    ax.set_title(title)
    ax.set_xlabel(None)
    ylab = "Net # of bikes\n" + r"($\Delta=$" + f"{max_avail - min_avail}" + " bikes)"
    ax.set_ylabel(ylab, rotation=0, horizontalalignment='right')
    
    return ax

def matspy(A, ax=None, figsize=(6, 6), **kwargs):
    from matplotlib.pyplot import spy
    
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    A_dense = A.todense()
    ax.imshow(A_dense, interpolation='none', cmap='Blues', **kwargs)
    return ax
