import cPickle as pickle
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from .utils import LabelBinning
from collections import Counter, defaultdict
import re

import warnings
warnings.filterwarnings("ignore")

def create_dummy(df, index, col_name, label_binariser=None, colnames=[]):
    '''Dummyfy variables.

       If there are any NAs, fill them with some temporary unique value.
       Afterwards, drop the corresponding column, and fill with NAs
       the initial rows containing NAs.
    '''
    v_unique = sorted(df.unique())
    na_indices = None
    if any(df.isnull()):
        ## fill in with some big value
        fill_na = df.dropna().max() * 1000 + 1
        na_indices = np.where(df.isnull())[0]
        df = df.fillna(fill_na)
    ## if there are only two unique values,
    ## the number of columns will be one,
    ## otherwise equal to the number of columns.
    n_unique = len(v_unique)
    n_cols = n_unique if n_unique > 2 else 1
    if colnames == []:
        colnames = map(lambda x : col_name + '_dummy_' + str(v_unique[x]), np.arange(n_cols))
    if label_binariser is None:
        label_binariser = LabelBinarizer()
        tmp_df = pd.DataFrame(label_binariser.fit_transform(df.values), columns=colnames, index=index)
    else:
        tmp_df = pd.DataFrame(label_binariser.transform(df.values), columns=colnames, index=index)
    if na_indices is not None:
        ## drop last column for NA class
        ## by default LabelBinarizer create columns for all unique values
        ## thus, one column will be redundant
        if col_name + '_dummy_nan' in tmp_df.columns:
            tmp_df = tmp_df.drop(col_name + '_dummy_nan', axis=1)
        ## fill with NAs rows
        tmp_df.iloc[na_indices, :] = np.nan
    ## drop last unnecessary column
    if n_cols > 2:
        tmp_df = tmp_df.drop(tmp_df.columns[-1:], axis=1)#col_name + '_dummy_nan', axis=1)
    return tmp_df, label_binariser, colnames

def binning(df,
            ignore_cols,
            drop_cols,
            n_bins,
            binning_dict=None,
            verbose=False):
    """
    Binning (discretisation) of numeric columns.

    Args:
      df : pandas data frame.
      ignore_cols : a list of columns that should be excluded when applying transformations.
      drop_cols : a list of columns to be dropped.
      n_bins : number of bins.
      binning_dict : dictionary. A general configuration file.
      verbose : boolean. Whether to print debug information.

    Returns:
      X : pre-processed df. Data frame of features.
    """
    ## reset index
    df = df.reset_index(drop=True)
    ## numerics dtypes
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if binning_dict is None:
        ## transformations dict
        binning_dict = dict()
    ## dropping columns
    df = df.drop(df.columns.intersection(drop_cols), axis=1)
    cols = binning_dict.setdefault('cols', df.columns.difference(ignore_cols))
    ## iterating over columns in order to apply transformations
    for c_ in filter(lambda x: ('dummy' not in x) and (x in df.columns), cols): ## not binning dummies
        ## create new dict for this column
        ## link to current dict
        c_dict = binning_dict.setdefault(c_, dict())
        ## if numeric
        if df[c_].dtype in numerics:
            if verbose:
                print("BINNING COLUMN : {}".format(c_))
                print("***" * 5)
            if c_dict != dict():
                lbl_bin = c_dict['binning']
                df.loc[np.where(df[c_].notnull())[0], c_ + '_BINNED'] = lbl_bin.transform(df[c_].dropna())
            else:
                lbl_bin = LabelBinning()
                df.loc[np.where(df[c_].notnull())[0], c_ + '_BINNED'] = lbl_bin.fit_transform(df[c_].dropna(), n_bins=n_bins)
            c_dict.setdefault('binning', lbl_bin)
    return df, binning_dict

def preprocess(df,
               ignore_cols,
               drop_cols,
               dummy_cols=[],
               dummy_config=None,
               fill_na=False,
               binning=False,
               transform_dict=None,
               verbose=False):
    """
    Pre-processing data for further modeling.
    Filling in NAs, digitising, dummyfying.

    Args:
      df : pandas data frame.
      ignore_cols : a list of columns that should be excluded when applying transformations.
      drop_cols : a list of columns to be dropped.
      dummy_cols : a list of numeric columns to be treated as non-numeric ones.
      dummy_config : either None or a tuple of (top-k values to be kept,
                                                percentage of values that top-k are taking).
      fill_na : boolean. If True, fill numeric columns with mean, others by sampling.
      transform_dict : dictionary. A general configuration file.
      verbose : boolean. Whether to print debug information.

    Returns:
      df : pre-processed df. Data frame of features.
    """
    ## reset index
    df = df.reset_index(drop=True)
    ## numerics dtypes
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if transform_dict is None:
        ## transformations dict
        transform_dict = dict()
    ## dropping columns
    df = df.drop(df.columns.intersection(drop_cols), axis=1)
    cols = transform_dict.setdefault('cols', df.columns.difference(ignore_cols))
    ## iterating over columns in order to apply transformations
    for c_ in cols:
        ## create new dict for this column
        ## link to current dict
        c_dict = transform_dict.setdefault(c_, dict())
        ## if numeric
        if (df[c_].dtype in numerics) and (c_ not in dummy_cols):
            if verbose:
                print("PROCESSING NUMERIC COLUMN : {}".format(c_))
                print("***" * 5)
            ## fill na by mean value
            if fill_na or ('fill_na' in c_dict):
                fill_val = df[c_].dropna().mean() if fill_na else c_dict['fill_na']
                df[c_] = df[c_].fillna(fill_val)
                c_dict.setdefault('fill_na', fill_val)
        else:
            if verbose:
                print("PROCESSING OBJECT COLUMN : {}".format(c_))
                print("***" * 5)
            ## modyfing objects
            if fill_na or ('fill_na' in c_dict):
                ## fill_na by sampling
                c_values = df[c_].dropna() if fill_na else c_dict['fill_na']
                na_indices = np.where(df[c_].isnull())[0]
                c_dict.setdefault('fill_na', c_values)
                df[c_] = (df[c_]
                          .fillna(dict(zip(na_indices,
                                           np.random.choice(c_values,
                                                            size=len(na_indices))))))
            # digitising and dummyfying (optional)
            ## column summary
            if 'dummy' in c_dict:
                lbl_bin, colnames, c_summ, codes = c_dict['dummy']
            else:
                lbl_bin = None
                colnames = []
                c_summ = df[c_].value_counts()
                codes = np.arange(start=1, stop=len(c_summ) + 1)
            ## if a value not present in c_summ.index, replace it with NA
            not_in_c_summ = filter(lambda x: x not in c_summ.index, df[c_].value_counts().index)
            if len(not_in_c_summ) > 0:
                df[c_] = df[c_].replace(not_in_c_summ, np.nan)
            if dummy_config is not None:
                ## want to create dummy and we do not have binariser
                ## in case of running on new data, dummy config can be any value
                ## only if there are more than 2 unique values
                ## parsing configuration
                max_top_k, prcnt, one_hot = dummy_config
                if (lbl_bin is None) and len(c_summ) > 2:
                    ## idx where cumsum exceeds percentage
                    threshold_idx = min(max_top_k,
                                        np.where((c_summ / sum(c_summ))
                                                 .cumsum() > prcnt)[0][0] + 1)
                    if verbose:
                        print("KEEPING TOP-{} VALUES OUT OF {}"
                              .format(threshold_idx, len(c_summ)))
                        print("***" * 5)
                    ## modifying new columns
                    ## in cases there are no codes
                    codes[(threshold_idx):] = 0
                ## replace by codes
                df[c_] = df[c_].map(dict(zip(c_summ.index.tolist(), codes)))
                if one_hot:
                    ## dummyfying
                    tmp_df, lbl_bin, colnames = create_dummy(df[c_], df.index, c_, lbl_bin, colnames)
                    ## drop old column
                    df = df.drop(c_, axis=1)
                    ## adding dummy columns
                    df = pd.concat([df, tmp_df], axis=1)
                    ## keeping config
            else:
                df[c_] = df[c_].replace(c_summ.index, codes)
            c_dict.setdefault('dummy', (lbl_bin, colnames, c_summ, codes))
    return df, transform_dict

def normalise(df,
              ignore_cols,
              drop_cols,
              scaler=None,
              transform_dict=None,
              verbose=False):
    """
    Normalising numeric columns.

    Args:
      df : pandas data frame.
      ignore_cols : a list of columns that should be excluded when applying transformations.
      drop_cols : a list of columns to be dropped.
      scaler : a string. Either 'standard', 'minmax', or None.
      transform_dict : dictionary. A general configuration file.
      verbose : boolean. Whether to print debug information.

    Returns:
      X : pre-processed df. Data frame of features.
    """
    ## reset index
    df = df.reset_index(drop=True)
    ## numerics dtypes
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if transform_dict is None:
        ## transformations dict
        transform_dict = dict()
    ## dropping columns
    df = df.drop(df.columns.intersection(drop_cols), axis=1)
    cols = transform_dict.setdefault('cols', df.columns.difference(ignore_cols))
    ## iterating over columns in order to apply transformations
    for c_ in filter(lambda x: ('dummy' not in x) and ('BINNED' not in x), cols): ## not normalising dummies
        ## create new dict for this column
        ## link to current dict
        c_dict = transform_dict.setdefault(c_, dict())
        ## if numeric
        if df[c_].dtype in numerics:
            if verbose:
                print("NORMALISING COLUMN : {}".format(c_))
                print("***" * 5)
            if ('scaler' in c_dict):
                scl = c_dict['scaler']
                df.loc[np.where(df[c_].notnull())[0], c_] = scl.transform(df[c_].dropna())
            elif scaler == 'standard':
                scl = StandardScaler()
                df.loc[np.where(df[c_].notnull())[0], c_] = scl.fit_transform(df[c_].dropna())
            elif scaler == 'minmax':
                scl = MinMaxScaler()
                df.loc[np.where(df[c_].notnull())[0], c_] = scl.fit_transform(df[c_].dropna())
            else:
                raise Exception("scaler must be one of ['standard', 'minmax']")
            c_dict.setdefault('scaler', scl)
    return df, transform_dict

def create_sets(path_to_file,
                sep,
                ignore_cols_dummy,
                ignore_cols_norm,
                ignore_cols_binning,
                drop_cols,
                dummy_cols=[],
                dummy_config=None,
                fill_na=False,
                scaler=None,
                n_bins=0,
                transform_dict=None,
                normalise_dict=None,
                binning_dict=None,
                test_size=0,
                trn_ids=None,
                tst_ids=None,
                seed=42,
                verbose=False):
    """
    Pre-processing data for further modeling.
    Filling in NAs, digitising, dummyfying.
    Splitting into train and test sets.
    Normalising data.

    Args:
      path_to_file : path to data_frame.
      sep : delimiter.
      ignore_cols_dummy : a list of columns that
                          should be excluded when dummyfying / filling in NAs.
      ignore_cols_norm : a list of columns that should be excluded when normalising.
      ignore_cols_binning : a list of columns that should be excluded when binning.
      drop_cols : a list of columns to be dropped.
      dummy_cols : a list of numeric columns to be treated as non-numeric ones.
      dummy_config : either None or a tuple of (top-k values to be kept,
                                                percentage of values that top-k are taking).
      fill_na : boolean. If True, fill numeric columns with mean, others by sampling.
      scaler : a string. Either 'standard', 'minmax', or None.
      n_bins : int. Number of bins.
      transform_dict : dictionary. A general configuration file for filling in NAs and dummies.
      normalise_dict : dictionary. A general configuration file for normalisation.
      binning_dict : dictionary. A general configuration file for binning.
      test_size : proportion of test set size. If less than 0, return full dataset.
      trn_ids : data frame. If given, merge on original data frame to get train split.
      tst_ids : data frame. If given, merge on original data frame to get test split.
      seed : random number when splitting into train and test sets.
      verbose : boolean. Whether to print debug information.

    Returns:
      (train_set, test_set, na_config_dict, norm_config_dict).
    """
    ## read in data
    df_usual = pd.read_csv(path_to_file, delimiter=sep)#, encoding='cp1251')
    if 'AGE' in df_usual.columns:
        ## remove weird age values
        df_usual = df_usual.drop(df_usual[df_usual.AGE < 18].index)
    if 'QTY_CHLD' in df_usual.columns:
        ## creating new features
        df_usual['HAS_CHLD'] = 1 * (df_usual['QTY_CHLD'] > 0)
    if 'CODE' in df_usual.columns:
        ## modifying code for transactions data
        df_usual['CODE'] = df_usual['CODE'].astype(object)
    ## fillin in na, dummyfy
    df_tmp, p_dict = preprocess(df_usual,
                                ignore_cols=ignore_cols_dummy,
                                drop_cols=drop_cols,
                                dummy_cols=dummy_cols,
                                dummy_config=dummy_config,
                                fill_na=fill_na,
                                transform_dict=transform_dict,
                                verbose=verbose)
    ## train/test/split
    if (test_size > 0) or ((trn_ids is not None) and (tst_ids is not None)):
        if test_size > 0:
            df_train, df_test = train_test_split(df_tmp,
                                                 test_size=test_size, random_state=seed,
                                                 stratify=df_tmp.label)
        else:
            df_train, df_test = (df_tmp.merge(trn_ids, on=trn_ids.columns.tolist()),
                                 df_tmp.merge(tst_ids, on=tst_ids.columns.tolist()))
        ## binning
        if n_bins > 1:
            df_train, b_dict = binning(df_train,
                                       ignore_cols=ignore_cols_binning,
                                       drop_cols=drop_cols,
                                       n_bins=n_bins,
                                       binning_dict=binning_dict,
                                       verbose=verbose)
            df_test, _ = binning(df_test,
                                 ignore_cols=ignore_cols_binning,
                                 drop_cols=drop_cols,
                                 n_bins=0,
                                 binning_dict=b_dict,
                                 verbose=verbose)
        else:
            b_dict = None
        ## normalisation
        df_train, n_dict = normalise(df_train,
                                     ignore_cols=ignore_cols_norm,
                                     drop_cols=drop_cols,
                                     scaler=scaler,
                                     transform_dict=normalise_dict,
                                     verbose=verbose)
        df_test, _ = normalise(df_test,
                                     ignore_cols=ignore_cols_norm,
                                     drop_cols=drop_cols,
                                     scaler=scaler,
                                     transform_dict=n_dict,
                                     verbose=verbose)
        return (df_train, df_test, p_dict, n_dict, b_dict)
    else:
        if n_bins > 1:
            df_tmp, b_dict = binning(df_tmp,
                                     ignore_cols=ignore_cols_binning,
                                     drop_cols=drop_cols,
                                     n_bins=n_bins,
                                     binning_dict=binning_dict,
                                     verbose=verbose)
        else:
            b_dict = None
        ## normalisation
        df_tmp, n_dict = normalise(df_tmp,
                                   ignore_cols=ignore_cols_norm,
                                   drop_cols=drop_cols,
                                   scaler=scaler,
                                   transform_dict=normalise_dict,
                                   verbose=verbose)
        return (df_tmp, None, p_dict, n_dict, b_dict)

def prepare_trns(trns_df,
                 ids_df,
                 verbose=False):
    """
    Create transactions in a format suitable for further modeling;
    i.e. each row corresponds to a unique single case.


    Args:
      trns_df : transactions data frame.
      ids_df : data frame with id columns.
      verbose : boolean. Whether to print debug information.

    Returns:
      numpy array of size (len(ids_df), max_n_trns, n_feats).
    """
    ## initial ids
    init_ids = ids_df.columns.tolist()
    ## id_cols that will be dropped later
    id_cols = init_ids + ['n_trns', 'ID']
    ## instead of multiple ids, create one ID to `rule them all`
    ids_df['ID'] = np.arange(len(ids_df))
    ## merge transactions data on ids
    trns_df = pd.merge(trns_df, ids_df, how='left', on=init_ids)
    ## unique ids
    uq_ids = trns_df.ID.unique()
    ## defining parameters of output
    n_feats = trns_df.shape[1] - len(id_cols)
    ## max number of transactions
    max_trns = trns_df.groupby('ID').size().max()
    ## initialise output
    out = np.zeros(shape=[len(ids_df), max_trns, n_feats])
    if verbose:
        print "GENERATING OUTPUT OF SIZE [{}, {}, {}]".format(*out.shape)
        print("***" * 5)
    for i in np.arange(len(ids_df)):
        if i in uq_ids:
            tmp = trns_df.loc[trns_df['ID'] == i, :].sort_values(by='n_trns', axis=0, ascending=False)
            out[i, :len(tmp), :] = tmp.drop(id_cols, axis=1).values
    return out
