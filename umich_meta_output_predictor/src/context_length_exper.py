from create_plots_with_zero_pred import compute_errors
from train import train_gpt2
from core import Config
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import logging
from dyn_models import generate_drone_sample, generate_lti_sample, generate_lti_sample_new_eig  # , generate_pendulum_sample
from tqdm import tqdm
import pickle


def generate_data(config):
    logger = logging.getLogger(__name__)
    config = Config()
    config.parse_args()
    print("Collecting data for", config.dataset_typ)
    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        samples = [] #make sure that train and val samples are different
        print("Generating", num_tasks, "samples for", name)
        for i in tqdm(range(num_tasks)):
            if config.dataset_typ == "drone":
                sim_obj, sample = generate_drone_sample(
                    config.n_positions, sigma_w=1e-1, sigma_v=1e-1, dt=1e-1)
            else:
                if name == "train":
                    fsim, sample = generate_lti_sample(config.dataset_typ,
                                                   config.num_traces[name],
                                                   config.n_positions,
                                                   config.nx, config.ny,
                                                   sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise)
                else:
                    fsim, sample = generate_lti_sample(config.dataset_typ,
                                                   config.num_traces[name],
                                                   config.n_positions,
                                                   config.nx, config.ny,
                                                   sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise)
                    # #save fsim to file
                    # os.makedirs("../data", exist_ok=True)
                    # with open(f"../data/{name}_{config.dataset_typ}_fsim_val.pkl", "wb") as f:
                    #     pickle.dump(fsim, f)
                    
                repeated_A = np.repeat(sample["A"][np.newaxis,:,:], config.num_traces[name], axis=0)
                sample["A"] = repeated_A

                repeated_C = np.repeat(sample["C"][np.newaxis,:,:], config.num_traces[name], axis=0)
                sample["C"] = repeated_C
                # print("shape of sample items:", {k: v.shape for k, v in sample.items()})
                # print("shape of sample A:", sample["A"].shape)
            samples.extend([{k: v[i] for k, v in sample.items()} for i in range(config.num_traces[name])])

        os.makedirs("../data", exist_ok=True)
        with open(f"../data/{name}_{config.dataset_typ}.pkl", "wb") as f:
            pickle.dump(samples, f)
        return None

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
        handles.extend(ax.plot(avg, label="Number of Training Systems: " + str(embed_dims[i]), linewidth=3, marker="o"))
        if shade:
            ax.fill_between(t, (avg - std)[1:], (avg + std)[1:], facecolor=handles[-1].get_color(), alpha=0.2)
        i += 1

    ax.legend(fontsize=30, loc=legend_loc)
    ax.set_xlabel("t", fontsize=30)
    ax.set_ylabel("Prediction Error", fontsize=30)
    ax.grid(which="both")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.title.set_text("Prediction Error vs Number of Training Systems")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/embed_dim" + ("-changing" if config.changing else ""))

def plot_train_time(times, embed_dims):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(embed_dims, times, label="Training Time", linewidth=3, marker="o")
    ax.legend(fontsize=30, loc="upper right")
    ax.set_xlabel("Number of Training Systems", fontsize=30)
    ax.set_ylabel("Training Time", fontsize=30)
    ax.grid(which="both")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.title.set_text("Training Time vs Number of Training Systems")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/embed_dim_times" + ("-changing" if config.changing else ""))


if __name__ == "__main__":
    config = Config()
    config.parse_args()
    train_steps = [240, 245, 250, 255, 260, 265, 270]
    errors = []
    times = []
    for train_step in train_steps:
        config.override("n_positions", train_step)
        config.override("ckpt_path", "../outputs/GPT2/240409_161242.3279c4/checkpoints/batch_size_" + str(config.batch_size) + "_con_len_" + str(config.n_positions) + "_step=" + str(config.train_steps) + ".ckpt")
        generate_data(config) #generate data for the new context length
        train_time = train_gpt2(config)
        times.append(train_time)
        # error, irr = compute_errors(config)
        # errors.append(error)
    #save errors and times to a file

    # plot_errors_emb(errors, train_steps)
    plot_train_time(times, train_steps)

    # # Assuming `errors` is your list of dictionaries
    # for error in errors:
    #     for key in error:
    #         # Convert numpy arrays to nested lists
    #         if isinstance(error[key], np.ndarray):
    #             error[key] = error[key].tolist()

    # with open('num_tasks_errors.json', 'w') as f:
    #     json.dump(errors, f)

    np.save("num_tasks_times.npy", times)