from afsrr import ANALYSIS_LOGS_DIR, EXPERIMENTS_LOGS_DIR

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

text_size = 20
titlesize = text_size
axes_font_size = text_size
grid_font_size = text_size
linewidth = 2
markersize = 15
plt.rcParams["figure.figsize"] = [21, 11]
plt.rcParams['axes.labelsize'] = axes_font_size
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = linewidth
plt.rcParams['lines.markersize'] = markersize
plt.rcParams['xtick.labelsize'] = text_size
plt.rcParams['ytick.labelsize'] = text_size
plt.rcParams['font.size'] = text_size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = text_size
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

logs_files = [
    glob.glob(
        os.path.join(
            EXPERIMENTS_LOGS_DIR,
            f'Test_JointRegressionModel_*2550_Beats*',
        )
    )[-1],
    glob.glob(
        os.path.join(
            EXPERIMENTS_LOGS_DIR,
            f'Test_JointRegressionModel_*5100_Beats*',
        )
    )[-1],
    glob.glob(
        os.path.join(
            EXPERIMENTS_LOGS_DIR,
            f'Test_JointRegressionModel_*15300_Beats*',
        )
    )[-1],
    glob.glob(
        os.path.join(
            EXPERIMENTS_LOGS_DIR,
            f'Test_JointRegressionModel_*25500_Beats*',
        )
    )[-1],
]
colors = (
    'firebrick',
    'darkgreen',
    'royalblue',
    'darkviolet',
)
legend = (
    f'SR 2550',
    f'SR 5100',
    f'SR 15300',
    f'SR 25500',
)

if __name__ == "__main__":
    errors = []
    stds = []
    naive_errors = []
    naive_errors_stds = []
    for f_p in logs_files:
        y_gt_p = os.path.join(f_p, glob.glob(os.path.join(f_p, 'y_gt*'))[0])
        y_pred_p = os.path.join(f_p, glob.glob(os.path.join(f_p, 'y_pred*'))[0])

        with open(y_gt_p, 'rb') as f:
            y_gt = pickle.load(f)

        with open(y_pred_p, 'rb') as f:
            y_pred = pickle.load(f)

        error = (
                np.abs((y_pred[:, :, -1, :] - y_gt[:, :, -1, :])) /
                np.abs(y_gt[:, :, -1, :])
        )
        smae = np.mean(error)
        smae_std = np.std(np.mean(error, axis=1))

        naive_error = (
                np.abs((np.mean(y_gt[:, :, -2, :])[None, None, None] - y_gt[:, :, -1, :])) /
                np.abs(y_gt[:, :, -1, :])
        )
        naive_errors.append(np.mean(naive_error))
        naive_errors_stds.append(np.std(naive_error))
        errors.append(smae)
        stds.append(smae_std)

    fig, ax = plt.subplots()
    for i in range(len(logs_files)):
        ax.bar(legend[i], errors[i], width=0.5, color=colors[i])
        ax.errorbar(legend[i], errors[i], yerr=stds[i], color=colors[i])

    ax.bar('Naive Predictor', np.mean(naive_errors), width=0.5, color='k')
    ax.errorbar('Naive Predictor', np.mean(naive_errors), yerr=naive_errors_stds[i], color='k')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('# of RR Intervals')

    ax.set_ylim([0, 0.4])
    ax.set_ylabel('MAPE')
    plt.savefig(
        fname=os.path.join(ANALYSIS_LOGS_DIR, 'regression_mape.pdf'),
        orientation='landscape',
        format='pdf',
        bbox_inches='tight',
    )
    plt.show()
