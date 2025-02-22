from afsrr import ANALYSIS_LOGS_DIR, EXPERIMENTS_LOGS_DIR

import os
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics


text_size = 20
plt.rcParams['axes.labelsize'] = text_size
plt.rcParams['axes.titlesize'] = text_size
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 15
plt.rcParams['xtick.labelsize'] = text_size
plt.rcParams['ytick.labelsize'] = text_size
plt.rcParams['font.size'] = text_size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = text_size


alpha = 0.05
beta = None
logs_files = [
    os.path.join(
        EXPERIMENTS_LOGS_DIR,
        'AF_Classifier_Test_NSR_Only_2550_Beats',
        f'results_{beta}_beta_{alpha}_alpha.pkl',
    ),
    os.path.join(
        EXPERIMENTS_LOGS_DIR,
        'AF_Classifier_Test_NSR_Only_5100_Beats',
        f'results_{beta}_beta_{alpha}_alpha.pkl',
    ),
    os.path.join(
        EXPERIMENTS_LOGS_DIR,
        'AF_Classifier_Test_NSR_Only_15300_Beats',
        f'results_{beta}_beta_{alpha}_alpha.pkl',
    ),
    os.path.join(
        EXPERIMENTS_LOGS_DIR,
        'AF_Classifier_Test_NSR_Only_25500_Beats',
        f'results_{beta}_beta_{alpha}_alpha.pkl',
    ),
]
if __name__ == "__main__":
    rocs = []
    aucs = []
    for f_path in logs_files:
        with open(f_path, 'rb') as f:
            results = pickle.load(f)
            y = results['y']
            y_hat = results['y_hat']
            fpr, tpr, _ = skmetrics.roc_curve(y, y_hat)
            auc = skmetrics.roc_auc_score(y, y_hat)
            rocs.append((fpr, tpr))
            aucs.append(auc)

    fig, ax = plt.subplots(figsize=(21, 11))
    colors = (
        'firebrick',
        'darkgreen',
        'royalblue',
        'darkviolet',
    )
    legend = (
        f'SR 2550 (AUC={aucs[0]:.3f})',
        f'SR 5100 (AUC={aucs[1]:.3f})',
        f'SR 15300 (AUC={aucs[2]:.3f})',
        f'SR 25500 (AUC={aucs[3]:.3f})',
    )
    for i in range(len(rocs)):
        ax.plot(rocs[i][0], rocs[i][1], color=colors[i], label=legend[i])

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.savefig(
        fname=os.path.join(ANALYSIS_LOGS_DIR, f'rocs_nsr_only.pdf'),
        orientation='landscape',
        format='pdf',
        bbox_inches='tight',
    )
    plt.show()





