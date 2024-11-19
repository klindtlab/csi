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


SAVE_PATH = 'MY/results'

target_names = ['class', 'multiple_objects', 'background', 'color', 'brighter',
                'darker', 'style', 'larger', 'smaller', 'object_blocking',
                'person_blocking', 'partial_view', 'pattern', 'pose', 'shape',
                'subcategory', 'texture']

models = {
    'Input': 'input',
    'Vit-b-16 (init)': 'vit_b_16_init',
    'ResNet50 (init)': 'resnet50_init',
    'Vit-b-16 (trained)': 'vit_b_16',
    'ResNet50 (trained)': 'resnet50',
}

data_mean = []
data_sem = []
data_p = []
index = []
for model in models:
    with open(os.path.join(SAVE_PATH, '%s_analysis.pkl' % models[model]), 'rb') as f:
        tmp = pickle.load(f)
        test = tmp['test']
    data_mean.append(test[:, :, 0].mean(1)[None])
    index.append(model)
    data_sem.append(test[:, :, 0].std(1)[None] / np.sqrt(test.shape[1]))
    data_p.append(ttest_1samp(test[:, :, 0], popmean=0.5, axis=1)[1][None])
data_mean = np.concatenate(data_mean, 0)
data_sem = np.concatenate(data_sem, 0)
data_p = np.concatenate(data_p, 0)

order = np.argsort(data_mean[-1])
data_mean = data_mean[:, order]
data_sem = data_sem[:, order]
data_p = data_p[:, order]
target_names_ = [target_names[o].replace('_', '\n') for o in order]

df_mean = pd.DataFrame(data=data_mean, columns=target_names_, index=index)


# Plot half 1
df_half1 = pd.DataFrame(data=data_mean[:, ::2], columns=target_names_[::2], index=index)
ax = df_half1.T.plot.bar(yerr=data_sem[:, ::2])
# Bonferroni adjustment (https://en.wikipedia.org/wiki/Bonferroni_correction)
alpha = 0.05 / data_p.size
ps = data_p[:, ::2]
for i in range(ps.shape[0]):
    for j in range(ps.shape[1]):
        if ps[i, j] < alpha:  # Check if the bar is significant
            x = j + i / 10 - .25  # Bar index on the x-axis
            y = data_mean[:, ::2][i, j] + data_sem[:, ::2][i, j] + 0.015  # Slightly above the error bar
            ax.text(x, y, '*')

plt.hlines(.5, -1, 17, zorder=-1, color='grey', label='chance level')
plt.legend(ncol=3)
ax.set_ylim(.4, 1)
plt.grid()
fig = plt.gcf()
fig.set_size_inches(.5 * np.array((20, 4)))
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'fig_imagenetx_half1_sign.pdf'))
plt.show()


# Plot half 2
df_half2 = pd.DataFrame(data=data_mean[:, 1::2], columns=target_names_[1::2], index=index)
ax = df_half2.T.plot.bar(yerr=data_sem[:, 1::2])
ps = data_p[:, 1::2]
for i in range(ps.shape[0]):
    for j in range(ps.shape[1]):
        if ps[i, j] < alpha:  # Check if the bar is significant
            x = j + i / 10 - .25  # Bar index on the x-axis
            y = data_mean[:, 1::2][i, j] + data_sem[:, 1::2][i, j] + 0.015  # Slightly above the error bar
            ax.text(x, y, '*')

plt.hlines(.5, -1, 17, zorder=-1, color='grey', label='chance level')
plt.legend(ncol=3)
ax.set_ylim(.4, 1)
plt.grid()
fig = plt.gcf()
fig.set_size_inches(.5 * np.array((20, 4)))
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'fig_imagenetx_half2_sign.pdf'))
plt.show()