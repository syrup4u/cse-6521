import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle

mpl.rcParams["font.size"] = 16

dataset1 = [
    [
        "setting1",
        10,
        [6, 9, 6, 10, 19, 8, 9, 6, 12, 6],
        ("simple_majority", 3, 1, "mlp", 0)
    ],
    [
        "setting2",
        10,
        [2, 1, 1, 2, 1, 3, 1, 1, 1, 1],
        ("simple_majority", 3, 1, "mlp_op", 3)
    ],
    [
        "setting3",
        10,
        [2, 1, 1, 1, 1, 1, 3, 1, 1, 3],
        ("simple_majority", 3, 1, "set_transformer", 3)
    ],
    [
        "setting4",
        10,
        [10, 9, 2, 6, 2, 4, 11, 1, 15, 2],
        ("simple_majority", 4, 1, "mlp", 0)
    ],
    [
        "setting5",
        10,
        [2, 1, 2, 1, 1, 1, 1, 3, 2, 1],
        ("simple_majority", 4, 1, "mlp_op", 3)
    ],
    [
        "setting6",
        10,
        [2, 2, 3, 24, 4, 2, 3, 1, 2, 1],
        ("simple_majority", 4, 1, "set_transformer", 3)
    ],
    [
        "setting7",
        8,
        [42, 40, 35, 20, 33, 17, 20, 22],
        ("simple_majority", 5, 1, "mlp", 0)
    ],
    [
        "setting8",
        10,
        [1, 6, 1, 1, 1, 6, 2, 5, 8, 7],
        ("simple_majority", 5, 1, "mlp_op", 3)
    ],
    [
        "setting9",
        9,
        [4, 14, 1, 2, 26, 6, 2, 6, 6],
        ("simple_majority", 5, 1, "set_transformer", 3)
    ]
]

# plot bar
def plot_bar(group_data, group_labels, x_labels, title, ylabel, filename, error_bars=None):
    gap = 0.2
    bar_width = (1.0 - gap) / len(group_data)
    x = np.arange(len(x_labels))

    plt.figure(figsize=(10, 6))
    for i, data in enumerate(group_data):
        plt.bar(x + i * bar_width, data, width=bar_width, label=group_labels[i])
        if error_bars:
            plt.errorbar(x + i * bar_width, data, yerr=error_bars[i], fmt='none', ecolor='black', capsize=5)
    plt.xticks(x + bar_width * (len(group_data) - 1) / 2, x_labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

def draw_my_dataset(my_dataset):
    offset = [0, 3, 6, 9, 12]
    tmp_labels = ["sm", "sm", "sm", "ac", "ac"]
    n_groups = 3
    group_data = []
    group_labels = []
    x_labels = [tmp_labels[ii] + "-" + str(my_dataset[i][3][1]) + "-" + str(my_dataset[i][3][2]) for ii, i in enumerate(offset)]
    group_data_success = []
    for i in range(n_groups):
        group_data.append([my_dataset[offset_idx + i][2] for offset_idx in offset])
        group_labels.append(my_dataset[offset[0] + i][3][3])
        group_data_success.append([my_dataset[offset_idx + i][1] for offset_idx in offset])
    ave_group_data = []
    err_group_data = []
    for data in group_data:
        ave_group_data.append([sum(x) / len(x) for x in data])
        err_group_data.append([np.std(x) for x in data])
    title = "Convergence Comparison"
    ylabel = "Epochs"
    filename = "convergence_comparison.png"
    plot_bar(ave_group_data, group_labels, x_labels, title, ylabel, filename, error_bars=err_group_data)

    title = "Success Comparison"
    ylabel = "Success Rate"
    filename_2 = "success_rate.png"
    group_data_success = [[x / 10 for x in group] for group in group_data_success]
    plot_bar(group_data_success, group_labels, x_labels, title, ylabel, filename_2)

def draw_from_pkl(fpath: str):
    with open(fpath, "rb") as f:
        dataset = pickle.load(f)
    n_groups = 3
    group_data = []
    group_labels = []
    offset = [i * n_groups for i in range(len(dataset) // n_groups)]
    
    x_labels = [str(dataset[i][3][1]) + "-" + str(dataset[i][3][2]) for i in offset]
    group_data_success = []
    for i in range(n_groups):
        group_data.append([dataset[offset_idx + i][2] for offset_idx in offset])
        group_labels.append(dataset[offset[0] + i][3][3])
        group_data_success.append([dataset[offset_idx + i][1] for offset_idx in offset])
    ave_group_data = []
    err_group_data = []
    for data in group_data:
        ave_group_data.append([sum(x) / len(x) for x in data])
        err_group_data.append([np.std(x) for x in data])
    title = f"Convergence Comparison ({dataset[0][3][0]})"
    ylabel = "Epochs"
    filename = "convergence_comparison.png"
    plot_bar(ave_group_data, group_labels, x_labels, title, ylabel, filename, error_bars=err_group_data)

    title = f"Success Comparison ({dataset[0][3][0]})"
    ylabel = "Success Rate"
    filename_2 = "success_rate.png"
    group_data_success = [[x / 10 for x in group] for group in group_data_success]
    plot_bar(group_data_success, group_labels, x_labels, title, ylabel, filename_2)

if __name__ == "__main__":
    with open("./results/logs/benchmark/benchmark_ac.pkl", "rb") as f:
        dataset = pickle.load(f)
    dataset1.extend(dataset)
    draw_my_dataset(dataset1)
    # draw_from_pkl("benchmark_ac.pkl")