import logging

from lightning.pytorch.strategies import SingleDeviceStrategy
import pytorch_lightning as pl
import torch

from core import Config, training
from models import GPT2
from datasources import FilterDataset, DataModuleWrapper
import os


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    config = Config()
    config.parse_args()

    # Specify datasets used
    model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                 n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)

    val_dset = FilterDataset(f"../data/val_{config.dataset_typ}.pkl", use_true_len=True) if os.path.exists(f"../data/val_{config.dataset_typ}.pkl") else None
    datamodule = DataModuleWrapper(FilterDataset(f"../data/train_{config.dataset_typ}.pkl"), val_dset)

    # Define model
    output_dir = training.setup_train(model)
    print(model)
    callbacks, loggers = training.get_callbacks_and_loggers(model, output_dir)
    print("ckpt_path", config.ckpt_path)
    ckpt_path = config.ckpt_path if config.ckpt_path != '' else None
    
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        logger=loggers,
        # gpus=torch.cuda.device_count(),
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=50,
        max_epochs=config.num_epochs
    )
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file {ckpt_path} does not exist.")
        # os.makedirs(ckpt_path, exist_ok=True)
        # Handle the situation, e.g., by aborting the program, loading a different checkpoint, etc. 
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)




