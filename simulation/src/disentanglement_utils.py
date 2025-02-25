"""
Module Name: disentanglement_utils.py
Author: Alice Bizeul, adapted from https://github.com/brendel-group/cl-ica
Description: Disentanglement evaluation scores such as R2 and MCC.
"""
from sklearn import metrics
from sklearn import linear_model
import torch
import numpy as np
import scipy as sp
from munkres import Munkres
from typing import Union
from typing_extensions import Literal

__Mode = Union[
    Literal["r2"], Literal["adjusted_r2"], Literal["pearson"], Literal["spearman"]
]

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

def _disentanglement(z, hz, mode: __Mode = "r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    assert mode in ("r2", "adjusted_r2", "pearson", "spearman", "mae", "accuracy")

    if mode == "r2":
        return metrics.r2_score(z, hz), None
    elif mode == "accuracy":
        return metrics.accuracy_score(z,hz), None
    elif mode == "adjusted_r2":
        r2 = metrics.r2_score(z, hz)
        n = z.shape[0]
        p = z.shape[1]
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2, None
    elif mode in ("spearman", "pearson"):
        dim = z.shape[-1]

        if mode == "spearman":
            raw_corr, pvalue = sp.stats.spearmanr(z, hz)
        else:
            raw_corr = np.corrcoef(z.T, hz.T)
        corr = raw_corr[:dim, dim:]

        if reorder:
            munk = Munkres()
            indexes = munk.compute(-np.absolute(corr))

            sort_idx = np.zeros(dim)
            hz_sort = np.zeros(z.shape)
            for i in range(dim):
                sort_idx[i] = indexes[i][1]
                hz_sort[:, i] = hz[:, indexes[i][1]]

            if mode == "spearman":
                raw_corr, pvalue = sp.stats.spearmanr(z, hz_sort)
            else:
                raw_corr = np.corrcoef(z.T, hz_sort.T)

            corr = raw_corr[:dim, dim:]
            return np.diag(np.abs(corr)).mean(), corr

    elif mode in ["mae"]:
        return metrics.mean_absolute_error(z,hz), None



def linear_disentanglement(z, hz, mode: __Mode = "r2", train_test_split=False):
    """Calculate disentanglement up to linear transformations.

    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        train_test_split: Use first half to train linear model, second half to test.
            Is only relevant if there are less samples then latent dimensions.
    """
    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    if train_test_split:
        n_train = len(z) // 2
        z_1 = z[:n_train]
        hz_1 = hz[:n_train]
        z_2 = z[n_train:]
        hz_2 = hz[n_train:]
    else:
        z_1 = z
        hz_1 = hz
        z_2 = z
        hz_2 = hz

    model = linear_model.LinearRegression()
    model.fit(hz_1, z_1)

    hz_2 = model.predict(hz_2)

    inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

    return inner_result, (z_2, hz_2)

def orthogonal_linear_disentanglement(z, hz, mode="mae", train_test_split=False, scaler=1.0):
    """
    Calculate disentanglement up to orthogonal linear transformations.

    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be 'r2', 'pearson', 'spearman'
        train_test_split: Use first half to train the model, second half to test.
        n_epochs: Number of epochs to train the orthogonal matrix.
        lr: Learning rate for gradient descent.

    Returns:
        inner_result: Disentanglement measure.
        (z_2, hz_2): Ground-truth and predicted latents after orthogonal transformation.
    """
    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(hz, z)
    coef = model.coef_.T
    
    _, singular_vals, _ = np.linalg.svd(coef)
    inner_result = _disentanglement((1/scaler)*np.ones(coef.shape[0]), singular_vals, mode=mode, reorder=False)

    return inner_result, None

def orthogonal_linear_disentanglement_perf(z, hz, mode="r2", train_test_split=False):
    """
    Calculate disentanglement up to orthogonal linear transformations.

    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be 'r2', 'pearson', 'spearman'
        train_test_split: Use first half to train the model, second half to test.
        n_epochs: Number of epochs to train the orthogonal matrix.
        lr: Learning rate for gradient descent.

    Returns:
        inner_result: Disentanglement measure.
        (z_2, hz_2): Ground-truth and predicted latents after orthogonal transformation.
    """
    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    # split z, hz to get train and test set for linear model
    if train_test_split:
        n_train = len(z) // 2
        z_1 = z[:n_train]
        hz_1 = hz[:n_train]
        z_2 = z[n_train:]
        hz_2 = hz[n_train:]
    else:
        z_1 = z
        hz_1 = hz
        z_2 = z
        hz_2 = hz

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(hz_1, z_1)
    hz_2 = model.predict(hz_2)
    inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

    return inner_result, None
