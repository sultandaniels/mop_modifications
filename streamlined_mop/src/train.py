import logging

import pytorch_lightning as pl

from core import Config, training
from models import GPT2
from datasources import FilterDataset, DataModuleWrapper
import os
import time
import pickle
from lightning.pytorch.loggers import WandbLogger


def train_gpt2(model, config, output_dir): #input emd_dim as a parameter for the embed dim experiment plots
    # a function to train GPT2 model
    logger = logging.getLogger(__name__)
    config.parse_args()
    print("batch_size:", config.batch_size)
    print("train_steps:", config.train_steps)
    print("context length:", config.n_positions)
    print("num_tasks:", config.num_tasks)

    # # Specify datasets used
    # model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
    #              n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)

    val_dset = FilterDataset(output_dir + f"/data/val_{config.dataset_typ}{config.C_dist}.pkl", use_true_len=True) if os.path.exists(output_dir + f"/data/val_{config.dataset_typ}{config.C_dist}.pkl") else None
    # raise Exception("Just checking FilterDataset")

    datamodule = DataModuleWrapper(FilterDataset(output_dir + f"/data/train_{config.dataset_typ}{config.C_dist}.pkl"), val_dset)

    # Define model
    # output_dir = training.setup_train(model)
    print("output_dir:", output_dir)
    callbacks, loggers = training.get_callbacks_and_loggers(output_dir, config.train_int)#, config.n_embd)
    ckpt_path = config.ckpt_path if config.ckpt_path != '' else None
    print("ckpt_path:", config.ckpt_path)
    
    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     callbacks=callbacks,
    #     logger=loggers,
    #     gradient_clip_algorithm=config.gradient_clip_algorithm,
    #     gradient_clip_val=config.gradient_clip_val,
    #     log_every_n_steps=50,
    #     max_epochs=config.num_epochs
    # )

    wandb_logger = WandbLogger(log_model="all")
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=50,
        max_epochs=config.num_epochs
    )
    # time how long it takes to train the model
    time_start = time.time()
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file {ckpt_path} does not exist.")
        # os.makedirs(ckpt_path, exist_ok=True)
        # Handle the situation, e.g., by aborting the program, loading a different checkpoint, etc. 
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    time_end = time.time()
    return time_end - time_start

if __name__ == '__main__':
    # #load the numpy folder in ../data and unpack the data.pkl file and show the keys
    # # Load the pickle file
    # with open('../data/numpy/data.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # # If the data is a dictionary, print its keys
    # if isinstance(data, dict):
    #     print("data[observation].shape:", data["observation"].shape)
    #     print("data[state].shape:", data["state"].shape)
    # else:
    #     print("The loaded data is not a dictionary.")

    # with open("../data/val_ypred.pkl", "rb") as f:
    #         entries = pickle.load(f)
    #         print("keys of entries:", entries[0].keys())
    #         print("len of entries:", len(entries))
    #         print("shape of all the values for each key in entries[0]", {k: v.shape for k, v in entries[0].items()})
    #         # print("shape of entries:", entries["observation"].shape)

    config = Config()
    model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                 n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
    train_gpt2(model, config)
    




