import logging

import pytorch_lightning as pl
import torch
import os

from core import Config, training
from models import GPT2
from dyn_models import apply_kf, generate_lti_sample, generate_changing_lti_sample, generate_drone_sample, \
    apply_ekf_drone
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
    for i in range(1000): #iterate over 1000
        if config.dataset_typ == "drone": #if the dataset type is drone
            sim_obj, entry = generate_drone_sample(config.n_positions) #generate drone sample
            us.append(entry["actions"]) #append the actions
        else:
            if config.changing: #if the dataset is changing
                sim_obj, entry = generate_changing_lti_sample(config.n_positions, config.nx, config.ny,
                                                              n_noise=config.n_noise) #generate changing lti sample
            else:
                sim_obj, entry = generate_lti_sample(config.dataset_typ, config.n_positions,
                                                     config.nx, config.ny, n_noise=config.n_noise) #generate lti sample
        ys.append(entry["obs"]) #append the observations
        sim_objs.append(sim_obj) #append the sim object
    ys = np.array(ys)
    us = np.array(us)

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
    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) #get the errors

    n_noise = config.n_noise
    if config.dataset_typ == "drone":
        preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])
    else:
        preds_kf = np.array(
            [apply_kf(fsim, _ys, sigma_w=fsim.sigma_w * np.sqrt(n_noise), sigma_v=fsim.sigma_v * np.sqrt(n_noise)) for
             fsim, _ys in zip(sim_objs, ys[:, :-1])])
    errs_kf = np.linalg.norm((ys - preds_kf), axis=-1)

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

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    plot_errs(names, err_lss, ax=ax, shade=config.dataset_typ != "drone")
    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/{config.dataset_typ}" + ("-changing" if config.changing else ""))




