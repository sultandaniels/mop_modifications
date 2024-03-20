from create_plots_with_zero_pred import compute_errors
from train import train_gpt2
from core import Config
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from context_length_exper import generate_data

def plot_errors_emb(errors, embed_dims, train_steps, legend_loc="upper right", ax=None, shade=True):
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    i = 0
    for err_ls in errors:
        # get the transformer errors and name from the err_ls dictionary
        name = "MOP"
        err_ls = err_ls[name]
        traj_errs = err_ls.sum(axis=-1)
        print(name, "{:.2f}".format(traj_errs.mean(axis=(0, 1))))

        t = np.arange(1, err_ls.shape[-1])
        avg, std = err_ls.mean(axis=(0, 1)), err_ls.std(axis=(0, 1))
        handles.extend(ax.plot(avg, label="batch_size: " + str(embed_dims[i]) + " train steps: " + str(train_steps[i]), linewidth=3, marker="o"))
        if shade:
            ax.fill_between(t, (avg - std)[1:], (avg + std)[1:], facecolor=handles[-1].get_color(), alpha=0.2)
        i += 1

    ax.legend(fontsize=30, loc=legend_loc)
    ax.set_xlabel("t", fontsize=30)
    ax.set_ylabel("Prediction Error", fontsize=30)
    ax.grid(which="both")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.title.set_text("Prediction Error vs Step for Different Batch Size and Train Steps")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/embed_dim" + ("-changing" if config.changing else ""))

# def plot_train_time(times, batch_sizes, train_steps, ax=None):
#     #make a bar graph of the training times
#     if ax is None:
#         fig = plt.figure(figsize=(15, 9))
#         ax = fig.add_subplot(111)
#     labels = ["batch_size:" + str(batch_sizes[i]) + " train steps:" + str(train_steps[i]) for i in range(len(batch_sizes))]
#     ax.bar(labels, times, color='b')
#     ax.set_xlabel("Batch Size", fontsize=30)
#     ax.set_ylabel("Training Time (s)", fontsize=30)
#     ax.grid(which="both")
#     ax.tick_params(axis='both', which='major', labelsize=30)
#     ax.tick_params(axis='both', which='minor', labelsize=20)
#     ax.title.set_text("Training Time vs Batch Size and Train Steps")
#     os.makedirs("../figures", exist_ok=True)
#     fig.savefig(f"../figures/batch_size_train_time" + ("-changing" if config.changing else ""))
    
def plot_train_time_two_param(times, context_lens, num_systems, ax=None):
    #plot the times vs the num_systems for each context length
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    for i in range(len(context_lens)):
        ax.plot(num_systems, times[i], label="context_len: " + str(context_lens[i]), linewidth=3, marker="o")
    ax.legend(fontsize=30, loc="upper right")
    ax.set_xlabel("Batch Size", fontsize=30)
    ax.set_ylabel("Training Time (s)", fontsize=30)
    ax.grid(which="both")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.title.set_text("Training Time vs Batch Size for Different Context Lengths")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/num_systems_train_time" + ("-changing" if config.changing else ""))
    return None


if __name__ == "__main__":
    config = Config()
    config.parse_args()
    context_lens = [260] #[50, 100, 200, 250, 300, 350, 400]
    num_systems = [2,8,16,24,28,32,38,29] #[20000, 25000,30000,35000,40000]
    errors = []
    times = np.zeros((len(context_lens), len(num_systems)))
    i = 0 #index for context_lens
    for context_len in context_lens:
        j = 0 #index for num_systems
        for num_system in num_systems:
            config.override("n_positions", context_len)
            config.override("batch_size", num_system)
            config.override("ckpt_path", "../outputs/GPT2/240319_062031.0ec795/checkpoints/batch_size_" +str(num_system) + "_con_len_" + str(context_len) + "_step=" + str(config.train_steps) + ".ckpt")
            generate_data(config) #generate data for the new context length and number of systems   
            train_time = train_gpt2(config)
            times[i,j] = train_time
            # error, irr = compute_errors(config)
            # errors.append(error)
            j += 1
        i += 1
    #save errors and times to a file

    # plot_errors_emb(errors, context_lens, num_systems)
    plot_train_time_two_param(times, context_lens, num_systems)

    # # Assuming `errors` is your list of dictionaries
    # for error in errors:
    #     for key in error:
    #         # Convert numpy arrays to nested lists
    #         if isinstance(error[key], np.ndarray):
    #             error[key] = error[key].tolist()

    # with open('num_tasks_context_len_errors.json', 'w') as f:
    #     json.dump(errors, f)

    np.save("batch_size_context_len_times.npy", times)