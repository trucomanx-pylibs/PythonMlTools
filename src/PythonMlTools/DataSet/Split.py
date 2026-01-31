#!/usr/bin/python

#import os
#import sys

import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split_stratify_index(y, test_size=0.38, random_state=42):
    """
    Split dataset indices into stratified train and test sets based on labels.

    This function returns the indices of the samples assigned to the training
    and test sets, while preserving the class distribution defined by `y`
    (stratified split). It is useful when the actual data arrays are large,
    stored on disk, or managed externally, and only indices are required.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels used for stratification.

    test_size : float, default=0.38
        Proportion of the dataset to include in the test split.
        Must be between 0.0 and 1.0.

    random_state : int, default=42
        Controls the randomness of the split for reproducibility.

    Returns
    -------
    idx_train : ndarray of shape (n_train_samples,)
        Indices of samples assigned to the training set.

    idx_test : ndarray of shape (n_test_samples,)
        Indices of samples assigned to the test set.

    y_train : ndarray of shape (n_train_samples,)
        Labels corresponding to `idx_train`.

    y_test : ndarray of shape (n_test_samples,)
        Labels corresponding to `idx_test`.

    Raises
    ------
    ValueError
        If `y` is not a 1-dimensional array.
    """

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1-dimensional array of labels")

    idx = np.arange(len(y))

    idx_train, idx_test, y_train, y_test = train_test_split(
        idx,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return idx_train, idx_test, y_train, y_test

