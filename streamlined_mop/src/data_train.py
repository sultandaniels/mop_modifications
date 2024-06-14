from collect_data import collect_data
from models import GPT2
from core import Config
from train import train_gpt2
from core import setup_train
import os
from create_plots_with_zero_pred import create_plots
import argparse
import wandb

# main function

if __name__ == '__main__':
    wandb.login()
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Predictions or not.')

    # Add the arguments
    parser.add_argument('--saved_preds', help='Boolean. Just plot the errors for a previously evaluated checkpoint', action='store_true')
    parser.add_argument('--make_preds', help='Boolean. Run predictions and plot the errors for a previously trained checkpoint', action='store_true')
    parser.add_argument('--resume_train', help='Boolean. Resume training from a specific checkpoint', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the flag
    print("saved preds arg", args.saved_preds)
    saved_preds = args.saved_preds
    print("make preds arg", args.make_preds)
    make_preds = args.make_preds
    print("resume train arg", args.resume_train)
    resume_train = args.resume_train


    config = Config() # create a config object
    # Get the class variables in dictionary format
    config_dict = config_dict = {
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

    if saved_preds:
        # create prediction plots
        run_preds = make_preds #run the predictions evaluation
        run_deg_kf_test = False #run degenerate KF test
        excess = False #run the excess plots
        shade = True
        config.override("ckpt_path", "/Users/sultandaniels/Documents/Transformer_Kalman/outputs/GPT2/240613_152144.dd8344_unifA_gauss_C/checkpoints/step=40000.ckpt")
        print("ckpt_path", config.ckpt_path)

        if resume_train:
            #get the parent directory of the ckpt_path
            parent_dir = os.path.dirname(config.ckpt_path)
            #get the parent directory of the parent directory
            output_dir = os.path.dirname(parent_dir)
            # instantiate gpt2 model
            model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                    n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
        
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

        create_plots(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade)
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

        # create prediction plots
        run_preds = True #run the predictions evaluation
        run_deg_kf_test = False #run degenerate KF test
        excess = False #run the excess plots
        shade = True

        print("ckpt_path", config.ckpt_path)
        create_plots(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade)