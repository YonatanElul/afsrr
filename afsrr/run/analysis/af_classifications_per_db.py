from afsrr import EXPERIMENTS_LOGS_DIR, ANALYSIS_LOGS_DIR

import os
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics


text_size = 18
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
colors = (
    'firebrick',
    'darkgreen',
    'royalblue',
    'darkviolet',
)
legend = (
    f'2550',
    f'5100',
    f'15300',
    f'25500',
)
db_names = (
    'LTAFDB',
    'AFDB',
    'Thew',
)
stats_per_db = {
    db_name: {
        model_name: None
        for model_name in legend
    }
    for db_name in db_names
}
afdb_starts = [0, 0, 0, 0, 0]
afdb_ends = [13, 13, 13, 13, 13]
ltafdb_starts = [13, 13, 13, 13, 13]
ltafdb_ends = [47, 40, 24, 21, 48]
thew_starts = [47, 40, 24, 21, 48]
thew_ends = [249, 241, 224, 221, 250]
rgird_angel = 45

if __name__ == "__main__":
    # Extract stats
    for model_i, f_path in enumerate(logs_files):
        with open(f_path, 'rb') as f:
            results = pickle.load(f)
            optimal_threshold = results['optimal_threshold']

        y = results['y']
        y_hat_af = (results['y_hat'].copy() >= optimal_threshold).astype(int)
        probs = results['y_hat']

        ltafdb_y = y[ltafdb_starts[model_i]:ltafdb_ends[model_i]]
        ltafdb_y_hat = y_hat_af[ltafdb_starts[model_i]:ltafdb_ends[model_i]]
        acc = sk_metrics.accuracy_score(ltafdb_y, ltafdb_y_hat)
        stats_per_db[db_names[0]][legend[model_i]] = acc

        afdb_y = y[afdb_starts[model_i]:afdb_ends[model_i]]
        afdb_y_hat = y_hat_af[afdb_starts[model_i]:afdb_ends[model_i]]
        acc = sk_metrics.accuracy_score(afdb_y, afdb_y_hat)
        stats_per_db[db_names[1]][legend[model_i]] = acc

        thew_y = y[thew_starts[model_i]:thew_ends[model_i]]
        thew_y_hat = y_hat_af[thew_starts[model_i]:thew_ends[model_i]]
        acc = sk_metrics.accuracy_score(thew_y, thew_y_hat)
        stats_per_db[db_names[2]][legend[model_i]] = acc

    # Plot radar per arrhythmia
    fig, axes = plt.subplots(ncols=len(db_names), sharey=True)
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

    for i, db_name in enumerate(stats_per_db):
        for model_i, model_name in enumerate(legend):
            axes[i].bar(model_name, stats_per_db[db_name][model_name], color=colors[model_i])

        axes[i].set_title(db_name)
        axes[i].set_xticks(list(range(len(legend))), legend, rotation=0)

    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[1].set_xlabel('# of RR Intervals')

    axes[0].set_ylabel('Classification Accuracy')
    plt.savefig(
        fname=os.path.join(ANALYSIS_LOGS_DIR, 'AccuracyPerDB.pdf'),
        orientation='landscape',
        format='pdf',
        bbox_inches='tight',
    )
    plt.show()
