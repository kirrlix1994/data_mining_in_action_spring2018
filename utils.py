import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score,auc, roc_curve, recall_score
from sklearn.utils import shuffle
from random import sample
import os
from collections import defaultdict


def KFoldTargetEncoding(x,
                         inner_splits,
                         group_col,
                         target_col,
                         n_col_name,
                         alpha,
                         noise_std):
    """KFold Target Encoding.
       For each fold, fill-in values in `group_col` using other folds.
       :math:`\frac{local\_mean \cdot nrows + global\_mean \cdot \alpha}{nrows + \alpha} + \mathcal{N}(0, std^{2})`


    Args:
      x : pandas data frame.
      inner_splits : list. Indices for each fold.
      group_col : str. Name of column for which the average target response
                       will be calculated.
      target_col : str. Name of target column.
      n_col_name : str. Name of new column.
      alpha : float. Regularisation parameter which regulates trade-off
                     between local (within-group) mean and global mean.
      noise_std: float. St. dev in `N(0, std)` noise.

    Returns:
      Pandas Series of the same length as `x` containing encoded target.
    """
    ## initialise new column
    x[n_col_name] = 0.0
    ## iterate over inner folds
    for j in range(len(inner_splits)):
        ## calculate new column values on all except for j
        fill_idx = inner_splits[j]
        ## at which idx to calculate
        calc_idx = np.concatenate(inner_splits[:j] + inner_splits[(j + 1):])

        x.loc[fill_idx, n_col_name] = targetEncoding(x.loc[calc_idx, [group_col, target_col]],
                                                     x.loc[fill_idx, [group_col]],
                                                     group_col,
                                                     target_col,
                                                     alpha,
                                                     noise_std)
    return x[n_col_name]

def targetEncoding(x_calc,
                   x_fill,
                   group_col,
                   target_col,
                   alpha,
                   noise_std):
    """Target Encoding.
       Fill-in values for values of `group_col` for `x_fill` from `x_calc`.
       :math:`\frac{local\_mean \cdot nrows + global\_mean \cdot \alpha}{nrows + \alpha} + \mathcal{N}(0, std^{2})`


    Args:
      x_calc : pd data frame. Used for calculating target statistics.
      x_fill : pd data frame. Used for filling-in statistics.
      group_col : str. Name of column for which the average target response
                       will be calculated.
      target_col : str. Name of target column.
      alpha : float. Regularisation parameter which regulates trade-off
                     between local (within-group) mean and global mean.
      noise_std: float. St. dev in `N(0, std)` noise.

    Returns:
      Pandas Series of the same length as `x_fill`.
    """
    ## global mean
    global_mean = x_calc[target_col].mean()
    ## dictionary: if key is not presented, replace by global mean
    calc_dict = defaultdict(lambda : global_mean)
    ## update dictionary
    calc_dict.update(x_calc
                     .groupby(group_col)
                     .apply(lambda x: (((np.mean(x[target_col]) * len(x)) +
                                        alpha * global_mean) /
                                        (len(x) + alpha)))
                     .to_dict())
    return (x_fill
            .loc[:, group_col]
            .apply(lambda x: calc_dict[x]) +
            np.random.normal(0, noise_std, size=len(x_fill))
           )

def HypeNKFoldCV(x,
                 group_cols,
                 target_col,
                 clf,
                 nfolds,
                 kfolds,
                 alpha,
                 noise_std,
                 scorer):
    """Hype NKFold Cross-Validation.
       Performs target encoding for each of `group_cols`,
       and evaluate the performance using two-staged folding.
       :math:`\frac{local\_mean \cdot nrows + global\_mean \cdot \alpha}{nrows + \alpha} + \mathcal{N}(0, std^{2})`

    Args:
      x : input data frame. Must contain all `group_cols` and `target_col`.
          During training, we use all columns but `target_col` for training.
      group_cols : list of str. Names of columns for which the average target response
                   will be calculated.
      target_col : str. Name of target column.
      clf : classifier object. Must have `fit` or `train` methods,
                               `predict` or `test` methods.
      nfolds : int. Number of outer folds.
      kfolds : int. Number of inner folds.
      alpha : float. Regularisation parameter which regulates trade-off
                     between local (within-group) mean and global mean.
      noise_std: float. St. dev in `N(0, std)` noise.
      scorer : function. Evaluation metric; must take two arguments:
               a vector of predictions and a vector of ground truth values.

    Returns:
      A list of `N` scores.
    """
    ## all indices
    all_idx = x.copy().index.values
    ## will shuffle indices for randomisation
    np.random.shuffle(all_idx)
    ## outer splits indices
    outer_splits = np.array_split(all_idx, nfolds)
    ## scorer results
    scores_val = []
    ## outer cycle
    for i in range(nfolds):
        ## keep `i`-th fold for validation
        val_idx = outer_splits[i]
        x_val = x.loc[val_idx].copy()
        ## choose all but `i`-th split
        inner_idx = np.concatenate(outer_splits[:i] + outer_splits[(i + 1):])
        ## further randomise training indices
        np.random.shuffle(inner_idx)
        ## split others further
        inner_splits = np.array_split(inner_idx, kfolds)
        ## training data frame
        x_train = x.loc[inner_idx].copy()
        ## iterate over group cols
        for group_col in group_cols:
            n_col_name = '_'.join([group_col, target_col])
            ## encode using division into KFolds
            x_train.loc[:, n_col_name] = KFoldTargetEncoding(x_train[[group_col, target_col]].copy(),
                                                             inner_splits,
                                                             group_col,
                                                             target_col,
                                                             n_col_name,
                                                             alpha,
                                                             noise_std)
            ## filling in the same column on val
            ## using whole `x_train`
            x_val.loc[:, n_col_name] = targetEncoding(x_train.loc[:, [group_col, target_col]],
                                                      x_val.loc[:, [group_col]],
                                                      group_col,
                                                      target_col,
                                                      alpha,
                                                      noise_std)

        ## will train on x_train
        ## will validate on x_val
        if 'fit' in dir(clf):
            clf.fit(x_train.drop(target_col, axis=1), x_train[target_col])
            preds_val = clf.predict(x_val.drop(target_col, axis=1))
        elif 'train' in dir(clf):
            clf.train(x_train.drop(target_col, axis=1), x_train[target_col])
            preds_val = clf.test(x_val.drop(target_col, axis=1)).argmax(axis=1)
        else:
            raise Exception("`clf` must contain either (`fit` and `predict`) or"
                            " (`train` and `test`) methods")
        scores_val.append(scorer(x_val[target_col], preds_val))
        del x_val, preds_val, x_train
    return scores_val

def leave_one_out(df, df_test=None, group_col='', target_col='', alpha=0.05, noise_std=0.0001):
    """Target encoding based on leave_one_out.
       https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions

    Args:
      df : data frame.
      df_test : test data frame.
      group_col : grouping column name.
      target_col : target column name.
      alpha : smoothing constant - trade-off between local and global means.
      noise_std : std of Gaussian noise added.

    Returns:
      Modified data frame with average target value per group column.
      mean + \alpha \cdot global_mean + noise
    """
    if df_test is None:
        ## keep only subset of data frame
        df_tmp = df[[group_col, target_col]]
        new_col_name = '_'.join([group_col, target_col])
        df_tmp[new_col_name] = 0.0
        for i in df_tmp.index:
            group_col_val = df_tmp.loc[i, group_col]
            other_vals = df_tmp.loc[(df_tmp.index != i), [target_col, group_col]]
            group_target = other_vals.loc[other_vals[group_col] == group_col_val, target_col]
            if len(group_target) == 0:
                group_mean = 0.0
            else:
                group_mean = group_target.mean()
            ## target vals mean + alpha * global_mean + small noise
            ## can connect noise to target_vals std
            df_tmp.loc[i, new_col_name] = (group_mean +
                                           other_vals[target_col].mean() * alpha +
                                           np.random.normal(0.0, scale=noise_std))
        return df_tmp[[new_col_name]]
    else:
        ## for test we will compute local means for each group
        ## if not found then use the global mean with noise
        ## keep only subset of data frame
        df_tmp = df[[group_col, target_col]]
        df_tmp_test = df_test[[group_col]]
        new_col_name = '_'.join([group_col, target_col])
        df_tmp_test[new_col_name] = 0.0
        ## pre-compute global mean
        global_mean = df_tmp[target_col].mean()
        for i in df_tmp_test.index:
            group_col_val = df_tmp_test.loc[i, group_col]
            train_col_val = df_tmp.loc[df_tmp[group_col] == group_col_val,
                                       target_col]
            if len(train_col_val) == 0:
                group_mean = 0.0
            else:
                group_mean = train_col_val.mean()
            ## target vals mean + alpha * global_mean + small noise
            ## can connect noise to target_vals std
            df_tmp_test.loc[i, new_col_name] = (group_mean +
                                                global_mean * alpha +
                                                np.random.normal(0.0, scale=0.0001))
        return df_tmp_test[[new_col_name]]

class LabelBinning():
    """Label Binning (Discretisation) of Continuous Variables."""
    def __init__(self):
        self.cuts = []
        self.labels = []
    def fit(self, x, n_bins):
        """
        Args:
          x : numeric data vector.
          n_bins : number of bins.
        """
        ## create equally spaced percentiles
        qs = 100 * np.linspace(start=1.0 / n_bins,
                               stop=1.0 - 1.0 / n_bins,
                               num=n_bins - 1)
        prcntls = np.unique(np.percentile(x, qs))
        ## add left and right infinity values
        self.cuts = np.insert(prcntls, [0, len(prcntls)], [-np.inf, np.inf])
        self.labels = np.arange(len(self.cuts) + 1)
    def fit_transform(self, x, n_bins):
        """
        Args:
          x : numeric data vector.
          n_bins : number of bins.

        Returns:
          Discretised x.
        """
        ## create equally spaced percentiles
        qs = 100 * np.linspace(start=1.0 / n_bins,
                               stop=1.0 - 1.0 / n_bins,
                               num=n_bins - 1)
        prcntls = np.unique(np.percentile(x, qs))
        ## add left and right infinity values
        self.cuts = np.insert(prcntls, [0, len(prcntls)], [-np.inf, np.inf])
        self.labels = np.arange(len(self.cuts) - 1)
        return pd.cut(x, self.cuts, labels=self.labels).astype(int)
    def transform(self, x):
        """
        Args:
          x : numeric data vector.

        Returns:
          Discretised x.
        """
        assert (self.cuts != []) & (self.labels != []), "Must call `fit` or `fit_transform` first"
        return pd.cut(x, self.cuts, labels=self.labels).astype(int)

def evaluate(probs, y_test, output_folder, file_prefix='test', model_names=None):
    """Plot ROC-curve. Find optimal threshold"""
    colours = ['b', 'g', 'm', 'c', 'y', 'r', 'k']

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    test_log = open(output_folder + '/' + file_prefix + '.log', 'w+')

    fprs, tprs, aucs = [], [], []
    for prob, model_name in zip(probs, model_names):
        test_log.write(model_name + "\n\n")
        pred = prob.argmax(axis=1)
        test_log.write(str(classification_report(y_test, pred)) + '\n')
        test_log.write('\n' + ' Predicted' + '\n')
        test_log.write(str(confusion_matrix(y_test, pred)) + '\n')

        fpr, tpr, thr = roc_curve(y_test, prob[:, 1])
        ## find best threshold : http://www.medicalbiostatistics.com/roccurve.pdf
        dist = np.sqrt((1. - tpr) ** 2 + (fpr) ** 2)
        best_thr = thr[np.argmin(dist)]
        best_thr_pred = (prob[:,1] > best_thr) * 1

        test_log.write('\n' + "Accuracy : " + str((accuracy_score(y_test, pred))) + '\n')
        test_log.write("F1 score : " + str(f1_score(y_test, pred)) + '\n')
        test_log.write("F1 score (thrs : {:.3f}) : ".format(best_thr) + str(f1_score(y_test, best_thr_pred)) + '\n')
        test_log.write("Recall : " + str(recall_score(y_test, pred)) + '\n')
        test_log.write("Precision : " + str(precision_score(y_test, pred)) + '\n\n')

        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

    if len(probs) > 1:
        model_names.extend(['mean', 'geom_mean'])
        test_log.write("Ensemble (mean)\n\n")
        prob = (np.array(probs).sum(axis=0) / 2)
        pred = prob.argmax(axis=1)
        test_log.write(str(classification_report(y_test, pred)) + '\n')
        test_log.write('\n' + ' Predicted' + '\n')
        test_log.write(str(confusion_matrix(y_test, pred)) + '\n')

        test_log.write('\n' + "Accuracy : " + str((accuracy_score(y_test, pred))) + '\n')
        test_log.write("F1 score : " + str(f1_score(y_test, pred)) + '\n')
        test_log.write("Recall : " + str(recall_score(y_test, pred)) + '\n')
        test_log.write("Precision : " + str(precision_score(y_test, pred)) + '\n\n')

        fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

        test_log.write("Ensemble (geom. mean)\n\n")
        prob = (np.array(probs).prod(axis=0) / np.array(probs).prod(axis=0).sum(axis=1)[:, np.newaxis])
        pred = prob.argmax(axis=1)
        test_log.write(str(classification_report(y_test, pred)) + '\n')
        test_log.write('\n' + ' Predicted' + '\n')
        test_log.write(str(confusion_matrix(y_test, pred)) + '\n')

        test_log.write('\n' + "Accuracy : " + str((accuracy_score(y_test, pred))) + '\n')
        test_log.write("F1 score : " + str(f1_score(y_test, pred)) + '\n')
        test_log.write("Recall : " + str(recall_score(y_test, pred)) + '\n')
        test_log.write("Precision : " + str(precision_score(y_test, pred)) + '\n\n')

        fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

    #plt.figure(figsize=(15, 15))
    for fpr, tpr, roc_auc, col, name in zip(fprs, tprs, aucs, colours, model_names):
        plt.plot(fpr, tpr, col, label='[%s] AUC = %0.5f' % (name, roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_folder + '/' + file_prefix + '_auc.png')
    plt.close()

    test_log.close()

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

## balanced dataset
class BalancedDataset():
    '''Generate batches.
    '''
    def __init__(self, X, y, batch_size=64):#X, y, batch_size=2):
        '''
        Args:
            X: feature matrix of size (n_samples, n_features);
            y: binary labels 0 and 1 of size (n_samples,);
            batch_size: total batch_size.
        '''

        self.n_features = X.shape[1]
        self.X = X.values
        self.y = y.values.astype(int)
        self.n_examples = y.shape[0]

        self.dict_examples = dict((k, shuffle(np.where(self.y == k)[0])) for k in np.unique(self.y))
        self.idx_examples = dict((k, 0) for k in np.unique(self.y))
        self.n_classes = len(self.idx_examples)
        self.batch_size = batch_size

    def _generate_batch(self):
        '''
        Generate batch of examples for a given class.

        Returns:
            Array of features with labels as the last column.
        '''
        arr_x = np.empty(shape=(self.batch_size, self.n_features))
        arr_y = np.empty(shape=(self.batch_size, ), dtype=np.int32)
        c_idx = 0 # current idx to fill
        need_examples = self.batch_size
        n_per_class = int(need_examples / self.n_classes)
        ## first take in each class as much as possible
        for i in range(self.n_classes):
            size_class = len(self.dict_examples[i])
            idx = self.idx_examples[i]
            n_ = min(idx + n_per_class, size_class)
            add_ = n_ - idx
            arr_x[c_idx : (c_idx + add_), :] = self.X[self.dict_examples[i][idx : n_], :]
            arr_y[c_idx : (c_idx + add_)] = self.y[self.dict_examples[i][idx : n_]]
            c_idx += add_
            if n_ == size_class: # nullify and shuffle if the end is reached
                self.idx_examples[i] = 0
                self.dict_examples[i] = shuffle(self.dict_examples[i])
            else:
                self.idx_examples[i] = n_
        ## if need_examples > 0 then we can take
        ## by one example from some classes
        #sh_classes = shuffle(range(self.n_classes))
        while True:
            if need_examples - c_idx > 0: ## need to add examples
                ## choose random class
                i = sample(range(self.n_classes), 1)[0]#sh_classes.pop()
                size_class = len(self.dict_examples[i])
                idx = self.idx_examples[i]
                n_ = min(idx + 1, size_class)
                add_ = n_ - idx
                arr_x[c_idx : (c_idx + add_)] = self.X[self.dict_examples[i][idx : n_], :]
                arr_y[c_idx : (c_idx + add_)] = self.y[self.dict_examples[i][idx : n_]]
                c_idx += add_
                if n_ == size_class: # nullify and shuffle if the end is reached
                    self.idx_examples[i] = 0
                    self.dict_examples[i] = shuffle(self.dict_examples[i])
                else:
                    self.idx_examples[i] = n_
            else:
                break
#         while True:
#             need_examples -= c_idx
#             if need_examples > 0: ## need to add examples
#                 ## choose random class
#                 i = sample(range(self.n_classes), 1)[0]
#                 size_class = len(self.dict_examples[i])
#                 idx = self.idx_examples[i]
#                 n_ = min(idx + need_examples, size_class)
#                 add_ = n_ - idx
#                 arr[c_idx : c_idx + add_] = self.y[self.dict_examples[i][idx : n_]]
#                 c_idx += add_
#                 if n_ == size_class: # nullify and shuffle if the end is reached
#                     self.idx_examples[i] = 0
#                     self.dict_examples[i] = shuffle(self.dict_examples[i])
#             else:
#                 break
        return (arr_x, arr_y)

    def next_batch(self):
        '''Generate next batch for training.'''
        batch_x, batch_y = self._generate_batch()
        batch_x, batch_y = shuffle(batch_x, batch_y)
        return batch_x, batch_y #[:, :-1], batch[:, -1]

class Dataset():
    '''Generate batches.
    '''
    def __init__(self, X, y, batch_size=2):
        '''
        Args:
            X: feature matrix of size (n_samples, n_features);
            y: binary labels 0 and 1 of size (n_samples,);
            batch_size: total batch_size.
        '''

        self.n_features = X.shape[1]
        self.n_examples = X.shape[0]

        self.Xy = np.concatenate([X, y[:, np.newaxis]], axis=1)

        ## initial shuffling
        np.random.shuffle(self.Xy)

        # indicators
        self.n_epochs = 0
        self.idx = 0

        self.batch_size = batch_size

    def _generate_batch(self):
        '''
        Generate batch of examples for a given class.

        Returns:
            Array of features with labels as the last column.
        '''
        arr = np.empty(shape=(self.batch_size, self.n_features + 1)) # n_features + 1 for label
        ## need to work with neg
        c_idx = 0 # current idx to fill
        need_examples = self.batch_size
        while True:
            ## first take as much as possible
            n_ = min(self.idx + need_examples, self.n_examples)
            add_ = n_ - self.idx ## how much else we need
            arr[c_idx : (c_idx + add_)] = self.Xy[self.idx : n_, :]
            # update current idx and the number of examples
            c_idx += add_
            need_examples -= add_
            if need_examples != 0:
                ## means that we reached the end
                self.idx = 0
                ## shuffle
                np.random.shuffle(self.Xy)
                self.n_epochs += 1
                #print("Epoch_neg #: %d" % self.epoch_neg)
            else:
                self.idx += add_
                break
        return arr

    def next_batch(self):
        '''Generate next batch for training.'''
        batch = self._generate_batch()
        np.random.shuffle(batch)
        return batch[:, :-1], batch[:, -1]

slim = tf.contrib.slim
def batch_normalisation(x, name, is_training, activation_fn=None, scale=True):
    with tf.variable_scope(name) as scope:
        output = slim.batch_norm(x,
                                 activation_fn=activation_fn,
                                 is_training=is_training,
                                 updates_collections=None,
                                 scale=scale,
                                 scope=scope)
    return output

class DataGenerator():
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be performed thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self,
                 data,
                 y,
                 max_seq_len=20, min_seq_len=1):
        self.data = []
        self.labels = y#pd.get_dummies(y).values
        self.seqlen = []
        for el in data:
            c_len = len(el)
            self.seqlen.append(c_len)
            if c_len < max_seq_len:
                # padding
                el = np.pad(el, [(0, max_seq_len - c_len), (0, 0)], 'constant')
            self.data.append(el)
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
            self.data, self.labels, self.seqlen = shuffle(self.data, self.labels, self.seqlen)
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

def dynamicRNN(x, seqlen, seq_max_len, n_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    multi_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_h) for n_h in n_hidden])

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(multi_cell, x, dtype=tf.float32,
                                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden[-1]]), index)

    # Linear activation, using outputs computed above
    return outputs #tf.matmul(outputs, weights['out']) + biases['out']

def save(saver, sess, logdir):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def report_softmax(gt, preds, n_classes, ignore_label=None):
    """
    Return accuracies and confusion matrix on multi-class predictions.
    If ignore_label is given, print also accuracy on all labels,
    except it.

    Args:
      gt: vector of ground truth labels.
      preds: array of predictions of size (len(gt), n_classes).
      n_classes: number of classes.
      ignore_label: if given, print also accuracy on all labels, except it.
    """
    df = pd.DataFrame(columns=['all_labels'] + range(n_classes))
    df.loc['recall', 'all_labels'] = np.mean(preds.argmax(axis=1) == gt)
    for i in range(n_classes):
        df.loc['recall', i] = np.mean(preds[gt == i].argmax(axis=1) == gt[gt == i]) # recall
        df.loc['precision', i] = np.mean(preds.argmax(axis=1)[preds.argmax(axis=1) == i]
                                == gt[preds.argmax(axis=1) == i]) # precision
        df.loc['fscore', i] = (2. * (df.loc['recall', i] * df.loc['precision', i]) /
                    (df.loc['recall', i] + df.loc['precision', i])) # f1-score
    df_cm = pd.DataFrame(confusion_matrix(gt, preds.argmax(axis=1)))
    if ignore_label is not None:
        df.loc['recall', 'no_' + str(ignore_label)] = (np.mean(preds[gt != ignore_label]
                                                .argmax(axis=1) ==
                                                    gt[gt != ignore_label]))
    return df, df_cm
