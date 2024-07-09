

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

from filterpy.kalman import KalmanFilter
from filterpy.common import Saver
from tqdm import tqdm

def kalmanFilterVL(sys, sigma_v, sigma_w):
    nx = sys["states"][0].shape[0]
    nz = sys["obs"][0].shape[0]
    W = np.eye(nx)*sigma_w*sigma_w
    V = np.eye(nz)*sigma_v*sigma_v

    kf = KalmanFilter(dim_x=nx, dim_z=nz)
    kf.x = np.zeros(sys["states"][0].shape)
    kf.P = solve_ricc(sys["A"], W)
    kf.Q = W
    kf.R = V
    kf.H = sys["C"]
    kf.F = sys["A"]

    return kf

def solve_ricc(A, W):  # solve the Riccati equation for the steady state solution
    L, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    Pi = (V @ (
            (Vinv @ W @ Vinv.T) / (1 - L[:, None] * L)
    ) @ V.T).real
    return Pi

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


    #######################

    preds_kf = np.array([[
            apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(config.n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(config.n_noise))
            for __ys in _ys
        ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
    ])  # get kalman filter predictions
    errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2

    ##############################

    # Discrete prior Kalman filter
    # Import systems
    # For each system:
    #   Apply Kalman -> normalization -> postprob -> dmmse
    #   Compute errors
    #   Save errors

    print("DMMSE predictions starting")
    t = time.time()
    with open(parent_parent_dir + f"/data/val_{config.dataset_typ}{config.C_dist}.pkl", "rb") as f:
        val_systems = pickle.load(f)
    
    with open(parent_parent_dir + f"/data/train_{config.dataset_typ}{config.C_dist}.pkl", "rb") as f:
        train_systems = pickle.load(f)
    
    distinct_train_systems = [train_systems[i] for i in range(0, len(train_systems), config.num_traces["train"])]

    preds_fpkf = []
    for i in range(config.num_val_tasks): # Is not being used, could be combined with below.
        for traceSys in tqdm(val_systems[i*config.num_traces["val"]:(i+1)*config.num_traces["val"]]):
            for filterSys in distinct_train_systems:
                kf = kalmanFilterVL(filterSys, config.sigma_v, config.sigma_w)
                xs, log_likelihood = [kf.x], [np.log(1/config.num_tasks)]

                for i, y in enumerate(traceSys["obs"][:-1,:]):
                    kf.update(y)
                    log_likelihood.append(kf.log_likelihood)
                    kf.predict()
                    xs.append(kf.x)

                filterSys["log_cl"] = np.cumsum(log_likelihood)
                filterSys["kf_pred"] = np.array([kf.H @ x for x in xs])
            
            normalization = np.max([sys["log_cl"] for sys in distinct_train_systems], axis=0)
            for sys in distinct_train_systems:
                sys["log_cl_norm"] = sys["log_cl"] - normalization
                sys["cl_norm"] = np.exp(sys["log_cl_norm"])
            total_likelihood_normalized = np.sum([sys["cl_norm"] for sys in distinct_train_systems], axis=0)
            for sys in distinct_train_systems:
                sys["postprob"] = sys["cl_norm"] / total_likelihood_normalized
        
            preds_fpkf.append(sum([sys["kf_pred"] * sys["postprob"][:,np.newaxis] for sys in distinct_train_systems]))

    preds_fpkf = np.reshape(preds_fpkf, (num_systems, num_trials, config.n_positions + 1, config.ny))
    errs_fpkf = np.linalg.norm((ys - preds_fpkf), axis=-1) ** 2
    print("DMMSE predictions done,", time.time() - t)
    # print(preds_fpkf.shape)
    # print(preds_fpkf)
    ##############################

    err_lss = collections.OrderedDict([
        ("Kalman", errs_kf),
        ("MOP", errs_tf),
        ("Zero", errs_zero),
        ("FPKF", errs_fpkf)
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


def load_preds(config):
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

    with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_err_lss.pkl", "rb") as f:
        err_lss_load = pickle.load(f)

    print("err_lss_load keys:", err_lss_load.keys())

    with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_irreducible_error.pkl", "rb") as f:
        irreducible_error_load = pickle.load(f)
    
    print(irreducible_error_load)
    
    return err_lss_load, irreducible_error_load


def make_plots(config, computed_preds=False):
    if not computed_preds:
        save_preds(config)
    err_lss, irreducible_error = load_preds(config)
    
    parent_dir = os.path.dirname(config.ckpt_path)
    print("parent_dir:", parent_dir)
    parent_parent_dir = os.path.dirname(parent_dir)
    os.makedirs(parent_parent_dir + "/figures", exist_ok=True)

    for i in range(config.num_val_tasks):
        fig, ax = plt.subplots(figsize=(7,5))

        for name, errs in err_lss.items():
            avg, std = errs[i,:,:].mean(axis=(0)), (3/np.sqrt(errs.shape[1]))*errs[i,:,:].std(axis=0)               
            ax.semilogy(avg, label=name)
            ax.fill_between(np.arange(config.n_positions + 1), avg + std, avg - std, alpha=0.2)

        fig.suptitle(f"Prediction errors for system {i}")
        ax.grid(True)
        ax.legend()
        fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}{config.C_dist}_system{i}")

if __name__ == "__main__":
    config = Config()
    
    num_tasks = 128
    config.override("ckpt_path", f"/home/jovyan/dd_outputs/240708_182434.1689c9_gaussA_gauss_C/{num_tasks}/checkpoints/step=65536.ckpt")
    config.override("num_tasks", num_tasks)
    config.override("num_traces", {"train": config.train_steps // num_tasks, "val": config.num_traces["val"]})

    make_plots(config, computed_preds=False)
    

    