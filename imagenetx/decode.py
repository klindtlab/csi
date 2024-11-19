import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr as corr
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from matplotlib import cm
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_1samp

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


SEED = 42
SAVE_PATH = 'MY/results'

target_names = ['class', 'multiple_objects', 'background', 'color', 'brighter',
                'darker', 'style', 'larger', 'smaller', 'object_blocking',
                'person_blocking', 'partial_view', 'pattern', 'pose', 'shape',
                'subcategory', 'texture']


# Function to balance a dataset
def balance_classes(X, y):
    # Separate the samples by class
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    # Determine the size of the smaller class
    n_samples = min(len(X_class_0), len(X_class_1))

    # Subsample both classes
    X_balanced_class_0 = X_class_0[:n_samples]
    X_balanced_class_1 = X_class_1[:n_samples]

    y_balanced_class_0 = np.zeros(n_samples, dtype=int)
    y_balanced_class_1 = np.ones(n_samples, dtype=int)

    # Combine the balanced classes
    X_balanced = np.vstack((X_balanced_class_0, X_balanced_class_1))
    y_balanced = np.concatenate((y_balanced_class_0, y_balanced_class_1))

    # Shuffle the balanced dataset
    X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=SEED)

    return X_balanced, y_balanced


def get_balanced_split(ind, seed=SEED, test_size=0.3, verbose=True):
    if ind == 0:
      y = results['targets'].copy()[:, ind] > np.mean(results['targets'].copy()[:, ind])
    else:
      y = results['targets'].copy()[:, ind]
    X_train, X_test, y_train, y_test = train_test_split(
        results['outputs'].copy(), y, test_size=test_size, random_state=seed, stratify=y)
    # Balance the training and test sets
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    X_test_balanced, y_test_balanced = balance_classes(X_test, y_test)

    # Print class distributions
    if verbose:
        print("Original class distribution in training set:", np.bincount(y_train.astype(int)))
        print("Balanced class distribution in training set:", np.bincount(y_train_balanced))
        print("Original class distribution in test set:", np.bincount(y_test.astype(int)))
        print("Balanced class distribution in test set:", np.bincount(y_test_balanced))
    return X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced


def eval(clf, x, y_true, verbose=False):
    # Assume you have true labels (y_true) and predicted labels (y_pred), and model probabilities (y_prob)
    y_pred = clf.predict(x)  # Predicted binary labels
    y_prob = clf.predict_proba(x)[:, 1:]  # Predicted probabilities

    # Accuracy, Correlation
    accuracy = np.mean(y_true == y_pred)
    correlation = corr(y_true, y_pred)[0]

    # Precision, Recall, F1
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    # Confusion Matrix
    if verbose:
      cm = confusion_matrix(y_true, y_pred)
      print(f"Confusion Matrix:\n{cm}")
    return accuracy, correlation, precision, recall, f1, roc_auc, pr_auc




models = ['resnet50', 'vit_b_16','resnet50_init', 'vit_b_16_init', 'input']

for m in models:
    print(m)
    with open(os.path.join(SAVE_PATH, '%s.pkl' % m), 'rb') as f:
        results = pickle.load(f)
    for k in results:
        print(k, results[k].shape, results[k].dtype)

    # filter out dead latents (for random resnet50) and zscore
    ind_good = results['outputs'].mean(0) > 0
    results['outputs'] = results['outputs'][:, ind_good]
    results['outputs'] -= np.mean(results['outputs'], 0, keepdims=True)
    results['outputs'] /= np.std(results['outputs'], 0, keepdims=True)

    X_train, y_train, X_test, y_test = get_balanced_split(4, seed=SEED, test_size=0.33)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    num_rep = 10
    train = np.zeros((len(target_names), num_rep, 7))
    test = np.zeros((len(target_names), num_rep, 7))
    metrics = ['accuracy', 'correlation', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    for i, name in enumerate(target_names):
        print(i, name)
        for rep in tqdm(range(num_rep)):
            X_train, y_train, X_test, y_test = get_balanced_split(
                i, seed=SEED + rep, test_size=0.3, verbose=rep==0)
            clf = LogisticRegression(random_state=SEED + rep)
            clf = clf.fit(X_train, y_train)
            train[i, rep] = np.array(eval(clf, X_train, y_train, verbose=False))
            test[i, rep] = np.array(eval(clf, X_test, y_test, verbose=False))
        print([f"train mean, {m}: {v:.4f}, " for m, v in zip(metrics, train[i].mean(0))])
        print([f"train std, {m}: {v:.4f}, " for m, v in zip(metrics, train[i].std(0))])
        print([f"test mean, {m}: {v:.4f}, " for m, v in zip(metrics, test[i].mean(0))])
        print([f"test std, {m}: {v:.4f}, " for m, v in zip(metrics, test[i].std(0))])

    with open('MY/results/%s_analysis.pkl' % m, 'wb') as f:
        pickle.dump({'train': train, 'test': test}, f)