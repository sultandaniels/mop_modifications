from collect_data import collect_data
from models import GPT2
from core import Config
from train import train_gpt2
from core import setup_train
import os
from create_plots_with_zero_pred import create_plots, convergence_plots
import argparse
import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from log_log_fit import loglogfit

def wandb_train(config_dict, model, output_dir):
    # add ckpt_path to config_dict
    config_dict["ckpt_path"] = config.ckpt_path

    # 🐝 1️⃣ Start a new run to track this script
    run = wandb.init(
        # Set the project where this run will be logged
        project="transformer_kalman_no_sweep",
        # Track hyperparameters and run metadata
        config=config_dict,
    )
    train_gpt2(model, config, output_dir) # train the model
    return None

def preds_thread(make_preds, resume_train, train_conv):
    # create prediction plots
    run_preds = make_preds # run the predictions evaluation
    run_deg_kf_test = False #run degenerate KF test
    excess = False #run the excess plots
    shade = True
    config.override("ckpt_path", "../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C/checkpoints/step=96000.ckpt")
    print("ckpt_path", config.ckpt_path)

    if resume_train:
        #get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)
        #get the parent directory of the parent directory
        output_dir = os.path.dirname(parent_dir)
        # instantiate gpt2 model
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
        
        wandb_train(config_dict, model, output_dir)
    if not train_conv:
        create_plots(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade)
    return run_preds, run_deg_kf_test, excess, shade


# main function

if __name__ == '__main__':
    wandb.login()
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Predictions or not.')

    # Add the arguments
    parser.add_argument('--saved_preds', help='Boolean. Just plot the errors for a previously evaluated checkpoint', action='store_true')
    parser.add_argument('--make_preds', help='Boolean. Run predictions and plot the errors for a previously trained checkpoint', action='store_true')
    parser.add_argument('--resume_train', help='Boolean. Resume training from a specific checkpoint', action='store_true')
    parser.add_argument('--train_conv', help='Boolean. make predictions for all checkpoints', action='store_true')
    parser.add_argument('--kfnorm', help='Boolean. subtract kalman performance from error', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the flag
    print("saved preds arg", args.saved_preds)
    saved_preds = args.saved_preds
    print("make preds arg", args.make_preds)
    make_preds = args.make_preds
    print("resume train arg", args.resume_train)
    resume_train = args.resume_train
    print("train conv arg", args.train_conv)
    train_conv = args.train_conv
    print("kfnorm arg", args.kfnorm)
    kfnorm = args.kfnorm


    config = Config() # create a config object
    # Get the class variables in dictionary format
    config_dict  = {
        "seed": 0,
        "fully_reproducible": False,
        "num_tasks": 40,
        "num_val_tasks": 3,
        "dataset_typ": "unifA",
        "C_dist": "_unif_C",
        "nx": 10,
        "ny": 5,
        "n_noise": 1,
        "num_traces": {"train": 1, "val": 2000},
        "train_steps": 7,
        "batch_size": 28,
        "train_data_workers": 1,
        "test_batch_size": 2,
        "test_data_workers": 1,
        "num_epochs": 1,
        "n_positions": 250,
        "n_embd": 128,
        "n_layer": 12,
        "n_head": 8,
        "n_dims_in": 5,
        "n_dims_out": 5,
        "changing": False,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "gradient_clip_algorithm": 'norm',
        "gradient_clip_val": 1.0
    }
    #change the num_tasks, num_val_tasks, dataset_typ, C_dist, nx, ny, n_noise, num_traces, train_steps, batch_size, train_data_workers, test_batch_size, test_data_workers, num_epochs, n_positions, n_embd, n_layer, n_head, n_dims_in, n_dims_out, changing, learning_rate, weight_decay, gradient_clip_algorithm, gradient_clip_val in config_dict to the values in config
    config_attributes = list(config_dict.keys())
    for key in config_attributes:
        config_dict[key] = config.__getattribute__(key)


    if (not train_conv) and (make_preds or saved_preds):
        run_preds, run_deg_kf_test, excess, shade = preds_thread(make_preds, resume_train, train_conv)
    elif train_conv:
        run_preds, run_deg_kf_test, excess, shade = preds_thread(make_preds, resume_train, train_conv)

        #for loop to iterate through all the checkpoints in the output directory
        output_dir = "../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C"
        fig, axs = plt.subplots(1, 3, figsize=(40, 20))  # 1 row, 3 columns, with a figure size of 15x5 inches
        filecount = 0

        sys_error_checkpoints_tuples = []
        ts = [50, 100, 200]
        for filename in os.listdir(output_dir + "/checkpoints/"):
            filecount += 1
            print("filecount:", filecount)
            config.override("ckpt_path", output_dir + "/checkpoints/" + filename)
            print("\n\n\nckpt_path", config.ckpt_path)
            step_avg_tup = convergence_plots(filecount, config, run_preds, run_deg_kf_test, kfnorm, config.num_val_tasks, shade, fig, axs, ts) #create the convergence plots and return the step and average error tuple
            sys_error_checkpoints_tuples.append(step_avg_tup) #append the tuple to the list of tuples

        #plot the error_checkpoints_tuples
        print("\n\nPlotting error_checkpoints_tuples")
        #make a new figure
        fig, ax = plt.subplots(3, 3, figsize=(30, 20))

        for sys in range(config.num_val_tasks):
            # Filter and transform sys_error_checkpoints_tuples for the current system sys
            error_checkpoints_tuples = [(str(x[0]), x[1][sys]) for x in sys_error_checkpoints_tuples if isinstance(x[1], list) and len(x[1]) > sys]
            # print("\nerror_checkpoints_tuples[0][1]", error_checkpoints_tuples[0][1])
            
            #sort the error_checkpoints_tuples by the step
            error_checkpoints_tuples = sorted(error_checkpoints_tuples, key=lambda x: int(x[0]))
        
            #make a plot for each value of t in ts for each system
            for t in range(len(ts)):

                x_values = [float(x[0]) for x in error_checkpoints_tuples]
                y_values = [x[1][t][0] for x in error_checkpoints_tuples]
                ax[t][sys].plot(x_values, y_values, marker='o', label="Median")

                print("y_values", y_values)
                
                # Fit a line to the data
                y_fit, m, c = loglogfit(x_values, y_values)

                ax[t][sys].plot(x_values, y_fit, label="Fit Line m = " + str(m) + " c = " + str(c))

                # Assuming the above prints confirm the lists are 1-dimensional
                y1 = [x[1][t][1] for x in error_checkpoints_tuples]
                y2 = [x[1][t][2] for x in error_checkpoints_tuples]
                x = np.arange(len(error_checkpoints_tuples))

                ax[t][sys].fill_between(x_values, y1, y2, alpha=0.2)
                ax[t][sys].set_title("System " + str(sys) + ": t = " + str(ts[t]) + (" Normalized" if kfnorm else ""))
                ax[t][sys].set_xlabel("Checkpoint Step")
                ax[t][sys].set_ylabel("Error")

                # Apply the formatter to the x-axis
                # ax[t][sys].xaxis.set_major_formatter(formatter)
                # ax[t][sys].legend()

                # # Rotate the x-axis labels
                # ax[t][sys].tick_params(axis='x', labelrotation=45)  # Rotate labels to 45 degrees
                # # Adjust the label size if necessary
                # ax[t][sys].tick_params(axis='x', labelsize=10)  # Adjust label size to 10 or any suitable size

                x_label_values = [int(x[0]) for x in error_checkpoints_tuples]
                # Assuming `x_label_values` is a list of values you want as labels on the x-axis
                x_label_positions = np.arange(len(x_label_values))  # Positions where labels should appear

                # Set the positions and labels for the x-axis ticks
                ax[t][sys].set_xticks(x_label_positions)
                ax[t][sys].set_xticklabels(x_label_values, rotation=45, fontsize=10)  # Rotate labels for better fit
                # set y-axis to log scale
                ax[t][sys].set_yscale('log')
                ax[t][sys].set_xscale('log')
                # add a legend 
                ax[t][sys].legend()

        fig.text(0.5, 0, "The error bars are the 45th and 55th percentile.", ha='center', va='bottom', fontsize=12)
        # Adjust layout to make room for the rotated x-axis labels
        plt.tight_layout()
        #get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        #get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + config.C_dist + "_system_conv_checks" + ("_normalized" if kfnorm else "") + ("-changing" if config.changing else ""))

    else:
        # instantiate gpt2 model
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                    n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
        
        output_dir = setup_train(model)
        # output_dir = output_dir + f"_{config.dataset_typ}{config.C_dist}"
        os.makedirs(output_dir + f"/data/", exist_ok=True)

        collect_data(model, config, output_dir) # collect data

        # replace ckpt_path with the path to the checkpoint file
        config.override("ckpt_path", output_dir + "/checkpoints/step=" + str(config.train_steps) + ".ckpt")

        wandb_train(config_dict, model, output_dir)

        # create prediction plots
        run_preds = True #run the predictions evaluation
        run_deg_kf_test = False #run degenerate KF test
        excess = False #run the excess plots
        shade = True

        print("ckpt_path", config.ckpt_path)
        create_plots(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade)
