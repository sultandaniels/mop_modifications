from dd_collect_data import collect_data
from models import GPT2
from core import Config
from train import train_gpt2
from core import setup_train
import os
from create_plots_with_zero_pred import create_plots, save_preds
import argparse
import wandb
import time
import hashlib

# main function
if __name__ == '__main__':
    wandb.login()

    # PARSER CAUSES ERROR, SO I COMMENTED IT OUT / Viktor
    # parser = argparse.ArgumentParser(description='Train')
    # parser.add_argument('--resume_train', help='Boolean.', action='store_true')
    # args = parser.parse_args()
    # resume_train = args.resume_train
    #################################################################################
    
    resume_train = True

    config = Config() 
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
    config_attributes = list(config_dict.keys())
    for key in config_attributes:
        config_dict[key] = config.__getattribute__(key)
    
    if resume_train:
        suite_identifier = "240708_110639.88723f_gaussA_gauss_C"
        output_dirs = {m: f"../dd_outputs/{suite_identifier}/{m}" for m in config.Ms}
        for _, output_dir in output_dirs.items():
            assert os.path.isdir(output_dir)
        config.parse_args()
    else:
        suite_identifier = time.strftime('%y%m%d_%H%M%S') + '.' + hashlib.md5(config.get_full_yaml().encode('utf-8')).hexdigest()[:6] +  f"_{config.dataset_typ}{config.C_dist}"
        output_dirs = {}

        for num_tasks in config.Ms:
            output_dirs[num_tasks] = f"../dd_outputs/{suite_identifier}/{num_tasks}"
            os.makedirs(output_dirs[num_tasks] + f"/data/", exist_ok=True)

        collect_data(config, output_dirs)

    for num_tasks, output_dir in output_dirs.items():
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
            n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)

        config.override("num_tasks", num_tasks)
        config_dict["num_tasks"] = config.num_tasks

        config.override("num_traces", {"train": config.train_steps // num_tasks, "val": config.num_traces["val"]})
        config_dict["num_traces"] = config.num_traces

        config.override("ckpt_path", output_dir + "/checkpoints/step=" + str(config.train_steps) + ".ckpt")
        config_dict["ckpt_path"] = config.ckpt_path

        run = wandb.init(
            # Set the project where this run will be logged
            project="Linear System Data Diversity",
            # Track hyperparameters and run metadata
            config=config_dict,
            reinit=True
        )
        train_gpt2(model, config, output_dir)
        run.finish()
