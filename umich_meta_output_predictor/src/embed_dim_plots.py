from create_plots_with_zero_pred import compute_errors
from train import train_gpt2
from core import Config
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def plot_errors_emb(errors, embed_dims, legend_loc="upper right", ax=None, shade=True):
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
        handles.extend(ax.plot(avg, label="embed_len: " + str(embed_dims[i]), linewidth=3))
        if shade:
            ax.fill_between(t, (avg - std)[1:], (avg + std)[1:], facecolor=handles[-1].get_color(), alpha=0.2)
        i += 1

    ax.legend(fontsize=30, loc=legend_loc)
    ax.set_xlabel("t", fontsize=30)
    ax.set_ylabel("Prediction Error", fontsize=30)
    ax.grid(which="both")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.title.set_text("Prediction Error vs Step for Different Embedding Lengths")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/embed_dim" + ("-changing" if config.changing else ""))

def plot_train_time(times, embed_dims):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(embed_dims, times, label="Training Time", linewidth=3)
    ax.legend(fontsize=30, loc="upper right")
    ax.set_xlabel("Embedding Length", fontsize=30)
    ax.set_ylabel("Training Time", fontsize=30)
    ax.grid(which="both")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.title.set_text("Training Time vs Embedding Length")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/embed_dim_times" + ("-changing" if config.changing else ""))


if __name__ == "__main__":
    config = Config()
    config.parse_args()
    embed_dims = [8, 16, 32, 64, 128, 256]
    errors = []
    times = []
    for embed_dim in embed_dims:
        config.override("n_embd", embed_dim)
        config.override("ckpt_path", "../outputs/GPT2/240313_224903.646dbe/checkpoints/emb_dim_" + str(embed_dim) + "_step=10000.ckpt")
        train_time = train_gpt2(config)
        times.append(train_time)
        error, irr = compute_errors(config, generate_data=True)
        errors.append(error)
    print(errors)
    #save errors and times to a file

    plot_errors_emb(errors, embed_dims)
    plot_train_time(times, embed_dims)

    # Assuming `errors` is your list of dictionaries
    for error in errors:
        for key in error:
            # Convert numpy arrays to nested lists
            if isinstance(error[key], np.ndarray):
                error[key] = error[key].tolist()

    with open('embed_dim_errors.json', 'w') as f:
        json.dump(errors, f)

    np.save("embed_dim_times.npy", times)
    
    
