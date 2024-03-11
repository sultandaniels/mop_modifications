import logging

import pytorch_lightning as pl
import torch
import os

from core import Config, training
from models import GPT2
from dyn_models import apply_kf_steady, generate_lti_sample, generate_lti_sample_steady, generate_changing_lti_sample, generate_drone_sample, apply_ekf_drone
#import the function solve_ricc from filtering_lti.py
from dyn_models.filtering_lti import solve_ricc
from datasources import FilterDataset, DataModuleWrapper
from utils import RLS, plot_errs
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu" #check if cuda is available
    logger = logging.getLogger(__name__)#get the logger
    config = Config() #get the config
    config.parse_args() #parse the arguments

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head).eval().to(device) #load_from_checkpoint

    ys, sim_objs, us = [], [], [] #initialize the lists
    num_trials = 5000
    for i in range(num_trials): #iterate over 1000 (I think this is the number of trials for the dataset)
        if config.dataset_typ == "drone": #if the dataset type is drone
            sim_obj, entry = generate_drone_sample(config.n_positions) #generate drone sample
            us.append(entry["actions"]) #append the actions
        else:
            if config.changing: #if the dataset is changing
                sim_obj, entry = generate_changing_lti_sample(config.n_positions, config.nx, config.ny,
                                                              n_noise=config.n_noise) #generate changing lti sample
            else:
                sim_obj, entry = generate_lti_sample_steady(config.dataset_typ, config.n_positions,
                                                     config.nx, config.ny, n_noise=config.n_noise) #generate lti sample
        ys.append(entry["obs"]) #append the observations
        sim_objs.append(sim_obj) #append the sim object
    ys = np.array(ys)
    us = np.array(us)
    print("ys.shape", ys.shape) #[num_trials, num_steps, num_output_dims]

    with torch.no_grad(): #no gradients
        I = ys[:, :-1] #get the inputs (observations without the last one)
        if config.dataset_typ == "drone": #if the dataset type is drone
            I = np.concatenate([I, us], axis=-1) #concatenate the inputs

        if config.changing:
            preds_tf = model.predict_ar(ys[:, :-1]) #predict using the model
        else:
            _, preds_tf = model.predict_step({"xs": torch.from_numpy(I).to(device)}) #predict using the model
            preds_tf = preds_tf["preds"].cpu().numpy() #get the predictions
            preds_tf = np.concatenate([np.zeros((preds_tf.shape[0], 1, preds_tf.shape[-1])), preds_tf], axis=1) #concatenate the predictions
    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) #get the errors of transformer predictions
    errs_zero = np.linalg.norm((ys - np.zeros(ys.shape)), axis=-1) #get the errors of zero predictions

    n_noise = config.n_noise
    if config.dataset_typ == "drone":
        preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])
    else:
        Pi = solve_ricc(sim_obj.A, (sim_obj.sigma_w**2)*np.eye(config.nx)) #set the initial state covariance matrix Pi to the steady state solution of the Riccati equation
        preds_kf = np.array(
            [apply_kf_steady(sim_obj, _ys, np.zeros(config.nx), Pi, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise)) for
             sim_obj, _ys in zip(sim_objs, ys[:, :-1])]) #get kalman filter predictions
    errs_kf = np.linalg.norm((ys - preds_kf), axis=-1)
    print("preds_kf.shape", preds_kf.shape) #(num_trials, num_steps, num_output_dims)
    #print first 10 kf predictions
    print("preds_kf[:10, 0, :]", preds_kf[:10, 0, :])

    err_lss = [errs_kf, errs_tf]
    names = ["Kalman", "MOP"]

    if config.dataset_typ != "drone":
        preds_rls = []
        for _ys in ys:
            ls = [np.zeros(config.ny)]
            rls = RLS(config.nx, config.ny)
            for i in range(len(_ys) - 1):
                if i < 2:
                    ls.append(_ys[i])
                else:
                    rls.add_data(_ys[i - 2:i].flatten(), _ys[i])
                    ls.append(rls.predict(_ys[i - 1:i + 1].flatten()))

            preds_rls.append(ls)
        preds_rls = np.array(preds_rls)
        errs_rls = np.linalg.norm(ys - preds_rls, axis=-1)
        err_lss.append(errs_rls)
        names.append("OLS")

    names.append("Zero") #append the name for zero predictions
    err_lss.append(errs_zero) #append the errors for zero predictions
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    plot_errs(names, err_lss, ax=ax, shade=config.dataset_typ != "drone")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/{config.dataset_typ}" + ("-changing" if config.changing else ""))
