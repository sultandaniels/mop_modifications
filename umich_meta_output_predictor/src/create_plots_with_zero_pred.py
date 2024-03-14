import logging

import collections
import pytorch_lightning as pl
import torch
import os

from core import Config, training
from models import GPT2
from dyn_models import apply_kf, generate_lti_sample, generate_changing_lti_sample, generate_drone_sample, \
    apply_ekf_drone
from dyn_models.filtering_lti import solve_ricc
from utils import RLS, plot_errs
import matplotlib.pyplot as plt
import numpy as np


def compute_errors(config):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger
    config.parse_args()  # parse the arguments

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head).eval().to(
        device)  # load_from_checkpoint

    ys, sim_objs, us = [], [], []  # initialize the lists
    num_systems = 1
    num_trials = 1000
    for i in range(num_systems):  # iterate over 1000 (I think this is the number of trials for the dataset)
        if config.dataset_typ == "drone":  # if the dataset type is drone
            sim_obj, entry = generate_drone_sample(config.n_positions)  # generate drone sample
            us.append(entry["actions"])  # append the actions
        else:
            if config.changing:  # if the dataset is changing
                sim_obj, entry = generate_changing_lti_sample(config.n_positions, config.nx, config.ny,
                                                              n_noise=config.n_noise)  # generate changing lti sample
            else:
                sim_obj, entry = generate_lti_sample(config.dataset_typ,
                                                     num_trials,
                                                     config.n_positions,
                                                     config.nx, config.ny,
                                                     n_noise=config.n_noise)  # generate lti sample
        ys.append(entry["obs"])  # append the observations
        sim_objs.append(sim_obj)  # append the sim object
    ys = np.array(ys)
    us = np.array(us)

    with torch.no_grad():  # no gradients
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)   # get the inputs (observations without the last one)
        if config.dataset_typ == "drone":  # if the dataset type is drone
            I = np.concatenate([I, us], axis=-1)  # concatenate the inputs

        if config.changing:
            preds_tf = model.predict_ar(ys[:, :-1])  # predict using the model
        else:
            batch_shape = I.shape[:-2]
            flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))

            _, flattened_preds_tf = model.predict_step({"xs": torch.from_numpy(flattened_I).to(device)})    # predict using the model
            preds_tf = np.reshape(flattened_preds_tf["preds"].cpu().numpy(), (*batch_shape, *I.shape[-2:])) # get the predictions
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf], axis=-2)  # concatenate the predictions

    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2     # get the errors of transformer predictions
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2     # get the errors of zero predictions

    n_noise = config.n_noise

    if config.dataset_typ == "drone":
        preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])
    else:
        preds_kf = np.array([[
                apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
                for __ys in _ys
            ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
        ])  # get kalman filter predictions
    errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2

    err_lss = collections.OrderedDict([
        ("Kalman", errs_kf),
        ("MOP", errs_tf),
        ("Zero", errs_zero)
    ])

    # if config.dataset_typ != "drone":
    #     preds_rls = []
    #     for _ys in ys:
    #         ls = [np.zeros(config.ny)]
    #         rls = RLS(config.nx, config.ny)
    #         for i in range(len(_ys) - 1):
    #             if i < 2:
    #                 ls.append(_ys[i])
    #             else:
    #                 rls.add_data_orig(_ys[i - 2:i].flatten(), _ys[i])
    #                 ls.append(rls.predict(_ys[i - 1:i + 1].flatten()))
    
    #         preds_rls.append(ls)
    #     preds_rls = np.array(preds_rls)
    #     err_lss["OLS"] = np.linalg.norm(ys - preds_rls, axis=-1) ** 2

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])

    return err_lss, irreducible_error

if __name__ == '__main__':
    config = Config()
    emb_dim = 256
    
    err_lss, irreducible_error = compute_errors(config, emb_dim)
    
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    plot_errs(err_lss, irreducible_error, ax=ax, shade=True, normalized=True)
    # plot_errs(err_lss, irreducible_error, ax=ax, shade=config.dataset_typ != "drone", normalized=True)

    # plot_errs(err_lss, irreducible_error, ax=ax, shade=True, normalized=False)
    # ax.plot(np.arange(config.n_positions + 1), np.full(config.n_positions + 1, np.mean(irreducible_error)), color='black', linewidth=5, linestyle='--')

    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/{config.dataset_typ}" + ("-changing" if config.changing else ""))
