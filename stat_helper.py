import numpy as np
import pandas as pd
import re
import SimpleITK as sitk
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from IPython import display

from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import scipy.stats
from scipy.stats import f_oneway


def compare_experiments(experiments, param, indices):
    results = []
    for idx in indices:
        print(np.mean(experiments[idx][param]))
        results.append(experiments[idx][param])

    return f_oneway(*results)


def res_class_dist(data, randseed):
    dataframes = []
    indices = list(data.keys())
    columns = ['0.5', '0.625', '1.0', '1.25']
    dataframe_labels = ['train', 'test']
    input_data = {}

    for dfl in dataframe_labels:
        input_data[dfl] = []
        for class_id in data:
            class_data = []
            for res_id in columns:
                class_data.append(data[class_id][dfl][res_id])
            input_data[dfl].append(class_data)

    for dfl in dataframe_labels:
        x = np.float32(np.array(input_data[dfl]))
        x /= np.expand_dims(np.sum(x, axis=1), 1)
        dataframes.append(
            pd.DataFrame(x,
                         index=indices,
                         columns=columns)
        )

    plot_clustered_stacked(dataframes, dataframe_labels,
                           title="Class Distribution against resolution ({0})".format(randseed))


# Statistical Variables
def get_mean(stat, skip=0, steps=3):

    if len(stat) < steps:
        return None

    x = list(reversed(stat))
    d = np.array(x[skip:skip + steps])
    val = d[:, 1]
    return sum(val) / steps


def get_fft_image(img):
    if type(img) is np.ndarray:
        img = sitk.GetImageFromArray(img)
    return sitk.ComplexToReal(sitk.FFTShift(sitk.ForwardFFT(sitk.FFTPad(img))))


def get_mean_diff(stat, steps=3):
    if len(stat) < steps * 2:
        return None

    x_new = get_mean(stat, skip=0, steps=steps)
    x_old = get_mean(stat, skip=steps, steps=steps)
    return x_new - x_old


def update_graph(loss_data, train_acc, test_acc, test_loss, learning_rate=None):

    display.clear_output(wait=True)

    plot_count = 2 if learning_rate is None else 3

    fig, ax_arr = plt.subplots(plot_count, sharex=True)

    d1 = np.array(loss_data)
    d2 = np.array(test_loss)
    x1_data = d1[:, 0]
    y1_data = d1[:, 1]
    x2_data = d2[:, 0]
    y2_data = d2[:, 1]
    ax_arr[0].set_ylabel('Loss')
    ax_arr[0].set_xlabel('Iterations')
    ax_arr[0].plot(x1_data, y1_data, 'r-', label="train loss")
    ax_arr[0].plot(x2_data, y2_data, 'g-', label="validation loss")
    ax_arr[0].legend(framealpha=0.5, loc="lower left")


    d1 = np.array(train_acc)
    d2 = np.array(test_acc)
    x1_data = d1[:, 0]
    y1_data = d1[:, 1]
    x2_data = d2[:, 0]
    y2_data = d2[:, 1]
    ax_arr[1].set_ylabel('Accuracy')
    ax_arr[1].set_xlabel('Iterations')
    ax_arr[1].plot(x1_data, y1_data, 'r-', label="training")
    ax_arr[1].plot(x2_data, y2_data, 'g-', label="validation")
    ax_arr[1].legend(framealpha=0.5, loc="lower left")

    if learning_rate is not None:
        d1 = np.array(learning_rate)
        x1_data = d1[:, 0]
        y1_data = d1[:, 1]
        ax_arr[2].set_ylabel('Learning rate')
        ax_arr[2].set_xlabel('Iterations')
        ax_arr[2].plot(x1_data, y1_data, 'b-', label="learning rate")
        ax_arr[2].legend(framealpha=0.5, loc="lower left")

    fig.set_size_inches(18.5, 5.25 * plot_count)

    plt.show()


def eval_experiments(path, iterations=20):

    import os
    from nn_model import NNModel
    experiments = {}
    for dr in os.listdir(path):
        full_path = path + dr
        dr_parts = dr.split('_')

        if len(dr_parts) != 8:
            continue
        [name, dataset, padding, classes, angles, fold, performance, date] = dr_parts

        experiment_id = '_'.join([name, classes, padding, angles])

        print(dr)
        for i in range(iterations):
            model = None
            try:
                model = NNModel(None, None, None, mode='test', model_state_path=full_path)
            except:
                continue

            scores = model.get_raw_eval()

            if experiment_id not in experiments:
                experiments[experiment_id] = {
                    'actual': [],
                    'original': [],
                    'predicted': [],
                    'actual_labels': scores['actual_labels'],
                    'original_labels': scores['original_labels'],
                    'class_map': scores['class_map'],
                    'accuracy': [],
                    'f1': [],
                    'precision': [],
                    'recall': [],
                }

            if scores['original'] is not None:
                experiments[experiment_id]['original'].extend(scores['original'])

            experiments[experiment_id]['actual'].extend(scores['actual'])
            experiments[experiment_id]['predicted'].extend(scores['predicted'])
            experiments[experiment_id]['accuracy'].append(accuracy_score(scores['actual'], scores['predicted']))
            experiments[experiment_id]['f1'].append(f1_score(scores['actual'], scores['predicted'], average='macro'))
            experiments[experiment_id]['precision'].append(precision_score(scores['actual'], scores['predicted'], average='macro'))
            experiments[experiment_id]['recall'].append(recall_score(scores['actual'], scores['predicted'], average='macro'))

    print("\n\n\n=============== Final report ===============")
    for exp_id in experiments:
        print_experiment(exp_id, experiments[exp_id])
    print("\n\n\n======================================")

    return experiments


def print_experiment(exp_id, exp):
    from nn_model import NNModel

    class_2_labels = ['Normal', 'Abnormal']
    class_6_labels = ['NN', 'EM', 'BV', 'GG', 'GR', 'HC']
    mapped_class_6_labels = ['NN', 'EM', 'BV', 'GG', 'GR', 'HC', 'MX']

    print("\n\n\n=============== " + exp_id + "  ===============")
    class_count = np.max(exp['actual']) + 1

    actual_labels = class_2_labels if class_count == 2 else class_6_labels
    mapped_labels = actual_labels if exp['class_map'] is None else mapped_class_6_labels

    class_report = classification_report(
        exp['actual'],
        exp['predicted'],
        target_names=actual_labels
    )

    mu = np.mean(exp['accuracy'])
    sigma = np.std(exp['accuracy'])

    print(scipy.stats.norm.interval(0.95, loc=mu, scale=sigma))
    print("μ = ",  str(mu))
    print("σ = ", str(sigma))

    print(class_report_latex(class_report))
    print(
        format_conf_matrix(
            actual_labels,
            actual_labels,
            NNModel.confusion_matrix(
                exp['actual_labels'],
                exp['actual_labels'],
                exp['actual'],
                exp['predicted']
            )
        )
    )

    if exp['class_map'] is not None:
        # print(exp['original'], exp['original_labels'])
        print(
            format_conf_matrix(
                actual_labels,
                mapped_labels,
                NNModel.confusion_matrix(
                    exp['actual_labels'],
                    exp['original_labels'],
                    exp['original'],
                    exp['predicted']
                )
            )
        )

def conf_matrix(class_labels_a, class_labels_p, actual, predicted):

    class_count_a = len(class_labels_a)
    class_count_p = len(class_labels_p)
    total_examples = len(predicted)
    row_count = class_count_a + 1
    col_count = class_count_p + 1
    matrix = [[0 for i in range(row_count)] for j in range(col_count)]
    for idx, val in enumerate(actual):
        a_i = actual[idx]
        p_i = predicted[idx]
        matrix[a_i][p_i] += 1
        matrix[-1][p_i] += 1
        matrix[a_i][-1] += 1
        matrix[-1][-1] += 1

    return matrix


def format_conf_matrix(class_labels_1, class_labels_2, matrix, fmt='latex'):

    body_col_count = len(class_labels_1) + 1
    body_row_count = len(class_labels_2) + 1

    heading = ['\diagbox{A}{P}'] + class_labels_1 + ['Total']

    template = """
\\begin{table}
\caption{\label{tab:table-name}Confusion matrix for }
\\begin{tabular}{r [TABLE_HEADING_HEAD]}
\\bottomrule
[TABLE_HEADING_BODY]
\\midrule
[TABLE_BODY]
\\midrule
[TABLE_TOTAL]
\\bottomrule
\end{tabular}
\end{table}
    """

    rows = []
    footer = ""
    lbls = class_labels_2 + ['Total']

    for idx_1, val in enumerate(matrix):
        for idx_2, val in enumerate(matrix[idx_1]):
            example_count = matrix[idx_1][-1]
            if example_count == 0:
                example_count = 1

            org = matrix[idx_1][idx_2]
            matrix[idx_1][idx_2] /= example_count
            matrix[idx_1][idx_2] *= 100
            matrix[idx_1][idx_2] = str(org) + " (" + str(int(round(matrix[idx_1][idx_2]))) + "\%)"

    for x, val_1 in enumerate(matrix):
        sv = [str(i) for i in val_1]
        if x == len(matrix) - 1:
            footer += lbls[x] + " & " + " & ".join(sv)
        else:
            if body_col_count == body_row_count:
                sv[x] = "\\textbf{" + sv[x] + "}"
            rows.append(lbls[x] + " & " + " & ".join(sv))

    data = {
        "TABLE_HEADING_HEAD": " c " * body_col_count,
        "TABLE_HEADING_BODY": " & ".join(heading) + "\\\\",
        "TABLE_BODY": " \\\\ \n".join(rows) + "\\\\",
        "TABLE_TOTAL": footer + "\\\\"
    }

    for idx in data:
        template = template.replace('[' + idx + ']', data[idx])
    return(template)

def class_report_latex(class_report):


    template_per = """
\\begin{table}
\caption{\label{tab:table-name}Performance metrics for }
\\begin{tabular}[[TABLE_HEAD1]]
\\bottomrule
[[TABLE_HEAD2]]
\\midrule
[[TABLE_BODY]]
\\midrule
[[TABLE_FOOTER]]
\\bottomrule
\end{tabular}
\end{table}

    """

    lines = class_report.split("\n")
    space_split_reg = re.compile("\s+")
    metrics = space_split_reg.split(lines[0])

    head1_row = "{"
    for m in metrics:
        head1_row += " c "
    head1_row += "}"

    head2_row = " & ".join(metrics) + "\\\\"

    per_string = template_per
    per_string = per_string .replace("[[TABLE_HEAD1]]", head1_row)
    per_string = per_string .replace("[[TABLE_HEAD2]]", head2_row)

    footer_data = lines[-2].replace("avg / total", "Avg/Total")
    footer_data = space_split_reg.split(footer_data)

    lines = lines[2:-2]
    body_rows = []
    for l in lines:
        row = space_split_reg.split(l.lstrip())
        body_rows.append(" & ".join(row))
    table_body = "\\\\\n".join(body_rows)
    per_string = per_string.replace("[[TABLE_BODY]]", table_body)

    table_footer = " & ".join(footer_data) + " \\\\\n"
    per_string = per_string.replace("[[TABLE_FOOTER]]", table_footer)




    return per_string + "\n\n\n"



def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    plt.tight_layout()
    return axe