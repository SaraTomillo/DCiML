import argparse
import traceback

import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.utils import indexable, Parallel, delayed
from sklearn.utils.validation import _num_samples

import estimation_methods
import IO
import time

import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")




def _enforce_prediction_order(classes, predictions, n_classes, method):
    """Ensure that prediction arrays have correct column order

    When doing cross-validation, if one or more classes are
    not present in the subset of data used for training,
    then the output prediction array might not have the same
    columns as other folds. Use the list of class names
    (assumed to be integers) to enforce the correct column order.

    Note that `classes` is the list of classes in this fold
    (a subset of the classes in the full training set)
    and `n_classes` is the number of classes in the full training set.
    """
    if n_classes != len(classes):
        recommendation = (
            'To fix this, use a cross-validation '
            'technique resulting in properly '
            'stratified folds')
        warnings.warn('Number of classes in training fold ({}) does '
                      'not match total number of classes ({}). '
                      'Results may not be appropriate for your use case. '
                      '{}'.format(len(classes), n_classes, recommendation),
                      RuntimeWarning)
        if method == 'decision_function':
            if (predictions.ndim == 2 and
                    predictions.shape[1] != len(classes)):
                # This handles the case when the shape of predictions
                # does not match the number of classes used to train
                # it with. This case is found when sklearn.svm.SVC is
                # set to `decision_function_shape='ovo'`.
                raise ValueError('Output shape {} of {} does not match '
                                 'number of classes ({}) in fold. '
                                 'Irregular decision_function outputs '
                                 'are not currently supported by '
                                 'cross_val_predict'.format(
                                    predictions.shape, method, len(classes)))
            if len(classes) <= 2:
                # In this special case, `predictions` contains a 1D array.
                raise ValueError('Only {} class/es in training fold, but {} '
                                 'in overall dataset. This '
                                 'is not supported for decision_function '
                                 'with imbalanced folds. {}'.format(
                                    len(classes), n_classes, recommendation))

        float_min = np.finfo(predictions.dtype).min
        default_values = {'decision_function': float_min,
                          'predict_log_proba': float_min,
                          'predict_proba': 0}
        predictions_for_all_classes = np.full((_num_samples(predictions),
                                               n_classes),
                                              default_values[method],
                                              dtype=predictions.dtype)
        predictions_for_all_classes[:, classes] = predictions
        predictions = predictions_for_all_classes
    return predictions


def _index_param_value(X, v, indices):
    """Private helper function for parameter value indexing."""
    if not _is_arraylike(v) or _num_samples(v) != _num_samples(X):
        # pass through: skip indexing
        return v
    if sp.issparse(v):
        v = v.tocsr()
    return safe_indexing(v, indices)

def _is_arraylike(x):
    """Returns whether the input is array-like"""
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))


def safe_indexing(X, indices):
    """Return items or rows from X using indices.

    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            warnings.warn("Copying input dataframe for slicing.",
                          DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]

def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels.

    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.

    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.

    Labels y will always be indexed only along the first axis.

    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.

    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.

    y : array-like, sparse matrix or iterable
        Targets to be indexed.

    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be use to slice the columns of X.

    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.

    y_subset : array-like, sparse matrix or list
        Indexed targets.

    """
    if getattr(estimator, "_pairwise", False):
        if not hasattr(X, "shape"):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = safe_indexing(X, indices)

    if y is not None:
        y_subset = safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset


def cross_val_predict(estimator, X, y=None, groups=None, cv='warn',
                      n_jobs=None, verbose=0, fit_params=None,
                      pre_dispatch='2*n_jobs', method='predict'):
    """Generate cross-validated estimates for each input data point

    The data is split according to the cv parameter. Each sample belongs
    to exactly one test set, and its prediction is computed with an
    estimator fitted on the corresponding training set.

    Passing these predictions into an evaluation metric may not be a valid
    way to measure generalization performance. Results can differ from
    `cross_validate` and `cross_val_score` unless all tests sets have equal
    size and the metric decomposes over samples.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" `cv` instance
        (e.g., `GroupKFold`).

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``

    See also
    --------
    cross_val_score : calculate score for each CV split

    cross_validate : calculate one or more scores and timings for each CV split

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.


    """
    start_time = time.time()
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    print("CV -> check_cv: %s seconds " % (time.time() - start_time))
    start_time = time.time()
    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = method in ['decision_function', 'predict_proba',
                        'predict_log_proba']
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=np.int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc

    print("CV -> encode: %s seconds " % (time.time() - start_time))

    start_time = time.time()
    """
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))
    """
    prediction_blocks = []
    for train, test in cv.split(X,y,groups):
        #prediction_blocks.append(_fit_and_predict(estimator, X, y, train, test, verbose, fit_params, method))
        fit_params = fit_params if fit_params is not None else {}
        fit_params = {k: _index_param_value(X, v, train)
                      for k, v in fit_params.items()}

        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, _ = _safe_split(estimator, X, y, test, train)

        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)
        func = getattr(estimator, method)
        predictions = func(X_test)
        if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
            if isinstance(predictions, list):
                predictions = [_enforce_prediction_order(
                    estimator.classes_[i_label], predictions[i_label],
                    n_classes=len(set(y[:, i_label])), method=method)
                    for i_label in range(len(predictions))]
            else:
                # A 2D y array should be a binary label indicator matrix
                n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
                predictions = _enforce_prediction_order(
                    estimator.classes_, predictions, n_classes, method)
        prediction_blocks.append((predictions, test))


    print("CV -> clone estimator and prediction blocks: %s seconds " % (time.time() - start_time))

    start_time = time.time()
    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])
    print("CV -> Concatenate the predictions: %s seconds " % (time.time() - start_time))

    start_time = time.time()
    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    print("CV -> check permutations: %s seconds " % (time.time() - start_time))
    start_time = time.time()
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)

        print("CV -> sp.issparse: %s seconds " % (time.time() - start_time))
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred

        print("CV -> concatenate method outputs for each label: %s seconds " % (time.time() - start_time))
    else:
        predictions = np.concatenate(predictions)

        print("CV -> concatenate predictions: %s seconds " % (time.time() - start_time))

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]



def error(predictions, y):
    return abs(predictions - y)

def regression(X_train, Y_train, X_test, Y_test):
    model = SVR(kernel='linear', C=1, coef0=0)

    start_time = time.time()
    model.fit(X_train, Y_train)
    print("MODEL FIT 1 : %s seconds " % (time.time() - start_time))
    start_time = time.time()
    Pred_Train = model.predict(X_train)
    Pred_Eval = model.predict(X_test)
    print("PREDICTIONS 1 : %s seconds " % (time.time() - start_time))

    eval = error(Pred_Eval, Y_test)
    start_time = time.time()
    CV = cross_val_predict(model, X_train, Y_train, cv=10)
    print("CROSS VAL : %s seconds " % (time.time() - start_time))
    CVError = error(CV, Y_train)

    importanceModel = SVR(kernel='linear', C=1, coef0=0)
    start_time = time.time()
    importanceModel.fit(X_train, Y_train)
    print("MODEL FIT 2 : %s seconds " % (time.time() - start_time))

    start_time = time.time()
    P_train = importanceModel.predict(X_train)
    P_test = importanceModel.predict(X_test)
    print("PREDICTIONS 2 : %s seconds " % (time.time() - start_time))
"""
    try:
        importances_kliep = estimation_methods.kliep(X_train, X_test)
        kliep_err = CVError * importances_kliep
    except Exception:
        print("Fallo kliep")
        kliep_err = []
        traceback.print_exc()

    try:
        importances_kliep_model = estimation_methods.kliep_model(P_train, P_test)
        kliep_model_err = CVError * importances_kliep_model
    except Exception:
        print("Fallo kliep con modelo")
        kliep_model_err = []
        traceback.print_exc()

    try:
        importances_kernel_density = estimation_methods.kernel_density(X_train, X_test, bandwith, kernel)
        kernel_density_err = CVError * importances_kernel_density
    except Exception:
        print("Fallo kernel density")
        kernel_density_err = []
        traceback.print_exc()

    try:
        importances_kernel_density_model = estimation_methods.kernel_density_model(P_train, P_test, bandwith, kernel)
        kernel_density_model_err = CVError * importances_kernel_density_model
    except Exception:
        print("Fallo kernel density con modelo")
        kernel_density_model_err = []
        traceback.print_exc()

    #return [eval, CVError, kliep_err, kliep_model_err, kernel_density_err, kernel_density_model_err]
    return [np.average(eval), np.average(CVError), np.average(kliep_err), np.average(kliep_model_err), np.average(kernel_density_err), np.average(kernel_density_model_err)]
"""


filename_train_red = "datasets-2032/wine-quality-red/wine-quality-red-train-0.33.csv"
filename_test_red = "datasets-2032/wine-quality-red/wine-quality-red-test-0.33-0.csv"

filename_train_white = "datasets-2032/wine-quality-white/wine-quality-white-train-0.33.csv"
filename_test_white = "datasets-2032/wine-quality-white/wine-quality-white-test-0.33-0.csv"

start_time = time.time()
#X_train, Y_train, X_test, Y_test = IO.readDataset(filename_train_red, filename_test_red)
X_train, Y_train, X_test, Y_test = IO.readDataset(filename_train_white, filename_test_white)
print("READING DATA : %s seconds " % (time.time() - start_time))
regression(X_train, Y_train, X_test, Y_test)
