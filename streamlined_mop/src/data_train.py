from collect_data import collect_data
from models import GPT2
from core import Config
from train import train_gpt2
from core import setup_train
import os

# main function

if __name__ == '__main__':
    config = Config() # create a config object
    # instantiate gpt2 model
    model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                 n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
    
    output_dir = setup_train(model)
    # output_dir = output_dir + f"_{config.dataset_typ}{config.C_dist}"
    os.makedirs(output_dir + f"/data/", exist_ok=True)

    collect_data(model, config, output_dir) # collect data

    # replace ckpt_path with the path to the checkpoint file
    config.override("ckpt_path", output_dir + "/step=" + str(config.train_steps) + ".ckpt")
    train_gpt2(model, config, output_dir) # train the model

