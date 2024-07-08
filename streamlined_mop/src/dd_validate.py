

import collections
import logging
import os
import copy
import pandas as pd
from pandas.plotting import table
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

from core import Config
from dyn_models import apply_kf, generate_lti_sample, generate_changing_lti_sample
from models import GPT2, CnnKF
from utils import RLS, plot_errs, plot_errs_conv
import pickle
import math
from tensordict import TensorDict


def compute_errors(config):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger

    num_systems = config.num_val_tasks  # number of validation tasks
    print("Number of validation systems:", num_systems)
    num_trials = config.num_traces["val"]  # number of traces
    print("Number of traces:", num_trials)

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head).eval().to(
        device)  # load_from_checkpoint

    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)
    print("parent_dir:", parent_dir)

    #get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    print("parent_parent_dir:", parent_parent_dir)

    with open(parent_parent_dir + f"/data/val_{config.dataset_typ}{config.C_dist}.pkl", "rb") as f:
        samples = pickle.load(f)
        # for every 2000 entries in samples, get the observation values and append them to the ys list
        i = 0
        ys = np.zeros((num_systems, num_trials, config.n_positions + 1, config.ny))
        for entry in samples:
            ys[math.floor(i/num_trials), i % num_trials] = entry["obs"]
            i += 1
        ys = ys.astype(np.float32)
        del samples  # Delete the variable
        gc.collect()  # Start the garbage collector

    #open fsim file
    with open(parent_parent_dir + f"/data/val_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "rb") as f:
        sim_objs = pickle.load(f)

    #Transformer Predictions
    start = time.time() #start the timer for transformer predictions
    with torch.no_grad():  # no gradients
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)   # get the inputs (observations without the last one)
        # if config.dataset_typ == "drone":  # if the dataset type is drone
        #     I = np.concatenate([I, us], axis=-1)  # concatenate the inputs

        if config.changing:
            preds_tf = model.predict_ar(ys[:, :-1])  # predict using the model
        else:
            # print("before model.predict_step()")
            batch_shape = I.shape[:-2]
            # print("batch_shape:", batch_shape)
            flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))
            # print("flattened_I.shape:", flattened_I.shape)
            validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I), batch_size=config.test_batch_size)
            preds_arr = [] # Store the predictions for all batches 
            for validation_batch in iter(validation_loader):
                _, flattened_preds_tf = model.predict_step({"xs": validation_batch.to(device)}) #.float().to(device)})    # predict using the model
                preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
            preds_tf = np.reshape(np.concatenate(preds_arr, axis=0), (*batch_shape, *I.shape[-2:])) # Combine the predictions for all batches
            # print("preds_tf.shape:", preds_tf.shape)
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf], axis=-2)  # concatenate the predictions
            # print("preds_tf.shape:", preds_tf.shape)
    end = time.time() #end the timer for transformer predictions
    print("time elapsed for MOP Pred:", (end - start)/60, "min") #print the time elapsed for transformer predictions

    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2     # get the errors of transformer predictions

    #zero predictor predictions
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2     # get the errors of zero predictions

    err_lss = collections.OrderedDict([
        # ("Kalman", errs_kf),
        ("MOP", errs_tf),
        ("Zero", errs_zero)
    ])

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    return err_lss, irreducible_error

def save_preds(config):
    err_lss, irreducible_error = compute_errors(config)  #, emb_dim)
    print("err_lss keys:", err_lss.keys())

    #make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)
    print("parent_dir:", parent_dir)

    #get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    print("parent_parent_dir:", parent_parent_dir)

    #get the step size from the ckpt_path
    step_size = config.ckpt_path.split("/")[-1].split("_")[-1]
    print("step_size:", step_size)

    os.makedirs(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size, exist_ok=True)
    #save err_lss and irreducible_error to a file
    with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_err_lss.pkl", "wb") as f:
        pickle.dump(err_lss, f)

    with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_irreducible_error.pkl", "wb") as f:
        pickle.dump(irreducible_error, f)
    
    return err_lss, irreducible_error


def make_plots(config):
    err_lss, irreducible_error = save_preds(config)
    errs = err_lss["MOP"]
    
    fig, ax = plt.subplots(1,config.num_val_tasks, figsize=(5*config.num_val_tasks,5))
    fig.suptitle("Prediction errors")
    for i in range(config.num_val_tasks):
        avg, std = errs[i,:,:].mean(axis=(0)), (3/np.sqrt(errs.shape[1]))*errs[i,:,:].std(axis=0)               
        ax[i].plot(avg)
    plt.show()

if __name__ == "__main__":
    config = Config()
    config.override("ckpt_path", "/home/viktor/Documents/Studier/Berkeley/dd_outputs/240707_155931.88723f_gaussA_gauss_C/4/checkpoints/step=10.ckpt")
    make_plots(config)
    

    