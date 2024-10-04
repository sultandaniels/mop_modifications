import collections
import copy
import gc
import logging
import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from tensordict import TensorDict
from pandas.plotting import table
from datetime import datetime

from core import Config
from dyn_models import apply_kf
from models import GPT2, CnnKF
from utils import RLS, plot_errs, plot_errs_conv

plt.rcParams['axes.titlesize'] = 20
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


####################################################################################################
# from wentinn's code
def wentinn_compute_errors(config):
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger
    config = Config()  # get the config
    config.parse_args()  # parse the arguments

    num_systems = config.num_tasks["test"]
    num_trials = config.num_traces["test"]

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head).eval().to(
        device)  # load_from_checkpoint

    with open(f"../data/test_sim.pt", "rb") as f:
        sim_objs = torch.load(f)
    with open(f"../data/test_{config.val_dataset_typ}.pt", "rb") as f:
        samples = torch.load(f)
        ys = samples["obs"].numpy()

    # ys, us = [], []  # initialize the lists
    # sim_objs = []
    # for i in range(num_systems):  # iterate over 1000 (I think this is the number of trials for the dataset)
    #     if config.dataset_typ == "drone":  # if the dataset type is drone
    #         sim_obj, entry = generate_drone_sample(config.n_positions)  # generate drone sample
    #         us.append(entry["actions"])  # append the actions
    #     else:
    #         if config.changing:  # if the dataset is changing
    #             sim_obj, entry = generate_changing_lti_sample(config.n_positions, config.nx, config.ny,
    #                                                           n_noise=config.n_noise)  # generate changing lti sample
    #         else:
    #             sim_obj, entry = generate_lti_sample(config.dataset_typ,
    #                                                  num_trials,
    #                                                  config.n_positions,
    #                                                  config.nx, config.ny,
    #                                                  n_noise=config.n_noise)  # generate lti sample
    #     ys.append(entry["obs"])  # append the observations
    #     sim_objs.append(sim_obj)  # append the sim object
    # ys = torch.stack(ys).numpy()
    # us = torch.stack(us).numpy()

    with torch.no_grad():  # no gradients
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)
        # if config.dataset_typ == "drone":  # if the dataset type is drone
        #     I = np.concatenate([I, us], axis=-1)  # concatenate the inputs

        if config.changing:
            preds_tf = model.predict_ar(ys[:, :-1])  # predict using the model
        else:
            batch_shape = I.shape[:-2]
            flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))

            _, flattened_preds_tf = model.predict_step(
                {"xs": torch.from_numpy(flattened_I).to(device)})  # predict using the model
            preds_tf = np.reshape(flattened_preds_tf["preds"].cpu().numpy(),
                                  (*batch_shape, *I.shape[-2:]))  # get the predictions
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                      axis=-2)  # concatenate the predictions

    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2  # get the errors of zero predictions

    n_noise = config.n_noise

    # if config.dataset_typ == "drone":
    #     preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])
    # else:
    #     preds_kf = np.array([[
    #             apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
    #             for __ys in _ys
    #         ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
    #     ])  # get kalman filter predictions
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

    if config.dataset_typ != "drone":
        #     preds_rls = []
        #     preds_rls_analytical = []
        #     for sim_obj, _ys in zip(sim_objs, ys):
        #         _preds_rls = []
        #         _preds_rls_analytical = []
        #         for __ys in _ys:3
        #             ls = [np.zeros(config.ny)]
        #             ls_analytical = [np.linalg.norm(__ys[0], axis=-1) ** 2]

        #             rls = RLS(config.nx, config.ny)
        #             for i in range(_ys.shape[-2] - 1):
        #                 if i < ir_length:
        #                     ls.append(__ys[i])
        #                     ls_analytical.append(np.linalg.norm(__ys[i + 1], axis=-1) ** 2)
        #                 else:
        #                     rls.add_data(__ys[i - ir_length:i].flatten(), __ys[i])
        #                     _cnn_rls = CnnKF(config.ny, ir_length)
        #                     _cnn_rls.observation_IR.data = torch.from_numpy(np.stack([_rls.mu for _rls in rls.rlss], axis=-1)
        #                                                                     .reshape(ir_length, config.ny, config.ny)
        #                                                                     .transpose(1, 0, 2)[:, ::-1].copy())

        #                     ls.append(rls.predict(__ys[i - ir_length + 1:i + 1].flatten()))
        #                     ls_analytical.append(_cnn_rls.analytical_error(sim_obj).item())

        #             _preds_rls.append(ls)
        #             _preds_rls_analytical.append(ls_analytical)

        #         preds_rls.append(_preds_rls)
        #         preds_rls_analytical.append(_preds_rls_analytical)

        #     err_lss["OLS"] = np.linalg.norm(ys - np.array(preds_rls), axis=-1) ** 2
        #     err_lss["OLS_analytical"] = np.array(preds_rls_analytical)

        # Debugging implemented OLS
        for ir_length in range(1, 4):
            print(f"IR length: {ir_length}")
            preds_rls_wentinn = []
            preds_rls_wentinn_analytical = []
            for sim_obj, _ys in zip(sim_objs, ys):
                _preds_rls_wentinn = []
                _preds_rls_wentinn_analytical = []
                for __ys in _ys:
                    padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])  # [(L + R - 1) x O_D]
                    ls = list(np.zeros((2, config.ny)))
                    ls_analytical = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)

                    rls_wentinn = CnnKF(config.ny, ir_length, ridge=1.0)
                    for i in range(config.n_positions - 1):
                        rls_wentinn.update(
                            torch.from_numpy(padded_ys[i:i + ir_length]),
                            torch.from_numpy(padded_ys[i + ir_length])
                        )

                        ls.append(rls_wentinn(torch.Tensor(padded_ys[i + 1:i + ir_length + 1])[None]).squeeze(0,
                                                                                                              1).detach().numpy())
                        ls_analytical.append(rls_wentinn.analytical_error(sim_obj).item())

                    _preds_rls_wentinn.append(ls)
                    _preds_rls_wentinn_analytical.append(ls_analytical)

                preds_rls_wentinn.append(_preds_rls_wentinn)
                preds_rls_wentinn_analytical.append(_preds_rls_wentinn_analytical)

            err_lss[f"OLS_ir_length{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
            # err_lss[f"OLS_ir_length{ir_length}_analytical"] = np.array(preds_rls_wentinn_analytical)

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    plot_errs(err_lss, irreducible_error, ax=ax, shade=True, normalized=False)
    # plot_errs(err_lss, irreducible_error, ax=ax, shade=config.dataset_typ != "drone", normalized=True)

    # plot_errs(err_lss, irreducible_error, ax=ax, shade=True, normalized=False)
    # ax.plot(np.arange(config.n_positions + 1), np.full(config.n_positions + 1, np.mean(irreducible_error)), color='black', linewidth=5, linestyle='--')

    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/{config.val_dataset_typ}" + ("-changing" if config.changing else ""))
    plt.show()


####################################################################################################
def compute_OLS_and_OLS_analytical(config, ys, sim_objs, ir_length, err_lss):  # PENDING DELETION
    preds_rls = []
    preds_rls_analytical = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _preds_rls = []
        _preds_rls_analytical = []
        for __ys in _ys:
            ls = [np.zeros(config.ny)]
            ls_analytical = [np.linalg.norm(__ys[0], axis=-1) ** 2]
            rls = RLS(config.nx, config.ny)

            # print("shape __ys:", __ys.shape)
            # print("range of for loop:", range(__ys.shape[-2] - 1))
            for i in range(_ys.shape[-2] - 1):
                if i < 3:
                    ls.append(__ys[i])
                    ls_analytical.append(np.linalg.norm(__ys[i + 1], axis=-1) ** 2)
                else:
                    if __ys[i - 3:i].shape[0] == 0:
                        print("i:", i)
                    rls.add_data(__ys[i - ir_length:i].flatten(), __ys[i])
                    _cnn_rls = CnnKF(config.ny, ir_length)
                    _cnn_rls.observation_IR.data = torch.from_numpy(np.stack([_rls.mu for _rls in rls.rlss], axis=-1)
                                                                    .reshape(ir_length, config.ny, config.ny)
                                                                    .transpose(1, 0, 2)[:, ::-1].copy())
                    ls.append(rls.predict(__ys[i - (ir_length - 1):i + 1].flatten()))
                    ls_analytical.append(_cnn_rls.analytical_error(sim_obj).item())
            _preds_rls.append(ls)
            _preds_rls_analytical.append(ls_analytical)

        preds_rls.append(_preds_rls)
        preds_rls_analytical.append(_preds_rls_analytical)

    err_lss["OLS"] = np.linalg.norm(ys - np.array(preds_rls), axis=-1) ** 2
    err_lss["OLS_analytical"] = np.array(preds_rls_analytical)
    return err_lss


def compute_OLS_and_OLS_analytical_revised(config, ys, sim_objs, ir_length, err_lss):
    # Ensure PyTorch version supports MPS and MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")  # Fallback to CPU if MPS is not available
    preds_rls = []
    preds_rls_analytical = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _preds_rls = []
        _preds_rls_analytical = []
        for __ys in _ys:
            # Convert numpy arrays to tensors and move to MPS
            __ys_tensor = torch.from_numpy(__ys).to(device)
            ls = [torch.zeros(config.ny, device=device)]
            ls_analytical = [torch.linalg.norm(__ys_tensor[0], axis=-1) ** 2]
            rls = RLS(config.nx, config.ny)  # Assuming RLS can handle MPS tensors
            for i in range(__ys_tensor.shape[-2] - 1):
                if i < 2:
                    ls.append(__ys_tensor[i])
                    ls_analytical.append(torch.linalg.norm(__ys_tensor[i + 1], axis=-1) ** 2)
                else:
                    rls.add_data_tensor(__ys_tensor[i - 2:i].flatten(), __ys_tensor[i])
                    _cnn_rls = CnnKF(config.ny, ir_length)
                    # Ensure _cnn_rls can handle MPS tensors
                    _cnn_rls.observation_IR.data = torch.stack([_rls.mu for _rls in rls.rlss], axis=-1).reshape(
                        ir_length, config.ny, config.ny).transpose(1, 0, 2)[:, ::-1].copy().to(device)
                    ls.append(rls.predict(__ys_tensor[i - 1:i + 1].flatten()))
                    ls_analytical.append(_cnn_rls.analytical_error(sim_obj).item())

            _preds_rls.append(ls)
            _preds_rls_analytical.append(ls_analytical)

        preds_rls.append(_preds_rls)
        preds_rls_analytical.append(_preds_rls_analytical)

    # Convert predictions back to CPU and numpy for final operations
    preds_rls = [torch.stack(pred).cpu().numpy() for pred in preds_rls]
    preds_rls_analytical = [torch.tensor(pred).cpu().numpy() for pred in preds_rls_analytical]

    err_lss["OLS"] = np.linalg.norm(ys - np.array(preds_rls), axis=-1) ** 2
    err_lss["OLS_analytical"] = np.array(preds_rls_analytical)
    return err_lss

def compute_OLS_ir(config, ys, sim_objs, max_ir_length, err_lss):

    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available

    # set torch precision to float64
    torch.set_default_dtype(torch.float64)
    print("max_ir_length + 1:", max_ir_length + 1)
    for ir_length in range(1, max_ir_length + 1):
        start = time.time()
        print(f"\tIR length: {ir_length}")

        if ir_length == 2:
            preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper(config, ys, sim_objs, ir_length, 0.0)

            err_lss[f"OLS_ir_{ir_length}_unreg"] = np.linalg.norm(ys - np.array(preds_rls_wentinn.cpu()), axis=-1) ** 2
            err_lss[f"OLS_analytical_ir_{ir_length}_unreg"] = np.array(preds_rls_wentinn_analytical.cpu())

            del preds_rls_wentinn
            del preds_rls_wentinn_analytical
            torch.cuda.empty_cache()
            gc.collect()

        preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper(config, ys, sim_objs, ir_length, 1.0)

        err_lss[f"OLS_ir_{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn.cpu()), axis=-1) ** 2
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.array(preds_rls_wentinn_analytical.cpu())
        end = time.time()
        print("\ttime elapsed:", (end - start) / 60, "min\n")

        del preds_rls_wentinn
        del preds_rls_wentinn_analytical   
        torch.cuda.empty_cache()
        gc.collect()

        # Check if CUDA is available
        if torch.cuda.is_available():

            # Print memory usage
            print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
            print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")
            print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 2):.2f} MB")
        else:
            print("CUDA is not available.")
    # set torch precision back to float32
    torch.set_default_dtype(torch.float32)
    return err_lss

def compute_OLS_helper(config, ys, sim_objs, ir_length, ridge):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available

    torch.set_default_device(device)
    preds_rls_wentinn = torch.zeros(np.expand_dims(ys[0],axis=0).shape)
    preds_rls_wentinn_analytical = torch.zeros(np.expand_dims(ys[0,:,:,0], axis=0).shape)


    with torch.no_grad():
        for sys in range(config.num_val_tasks):
            ys_sys = np.expand_dims(ys[sys], axis=0)
        
            # [n_systems x n_traces x (n_positions + 1) x O_D]
            torch_ys = torch.Tensor(ys_sys).to(device)


            # [n_systems x n_traces x (n_positions + ir_length) x O_D]
            padded_ys = torch.cat([
                torch.zeros((*torch_ys.shape[:2], ir_length - 1, config.ny)).to(device), torch_ys.to(device)
            ], dim=-2)

            Y_indices = torch.arange(ir_length, (config.n_positions - 1) + ir_length)[:, None]  # [(n_positions - 1) x 1]
            X_indices = Y_indices - 1 - torch.arange(ir_length)

            del torch_ys
            torch.cuda.empty_cache()
            gc.collect()

            X, Y = padded_ys[..., X_indices, :], padded_ys[..., Y_indices, :]   # [n_systems x n_traces x (n_positions - 1) x ir_length x O_D], [n_systems x n_traces x (n_positions - 1) x 1 x O_D]

            flattened_X, flattened_Y = X.flatten(-2, -1), Y.flatten(-2, -1)     # [n_systems x n_traces x (n_positions - 1) x (ir_length * O_D)], [n_systems x n_traces x (n_positions - 1) x O_D]

            # [n_systems x n_traces x (n_positions - 1) x (ir_length * I_D) x (ir_length * O_D)]
            cumulative_XTX = torch.cumsum(flattened_X[..., :, None].to(device) * flattened_X[..., None, :], dim=-3).to(device) + ridge * torch.eye(ir_length * config.ny).to(device)
            # [n_systems x n_traces x (n_positions - 1) x (ir_length * I_D) x O_D]
            cumulative_XTY = torch.cumsum(flattened_X[..., :, None] * flattened_Y[..., None, :], dim=-3)

            min_eqs = config.ny if ridge == 0.0 else 1

            _rank_full = torch.inverse(cumulative_XTX[..., min_eqs - 1:, :, :]) @ cumulative_XTY[..., min_eqs - 1:, :, :]
            _rank_deficient = []
            for n_eqs in range(1, min_eqs):
                _rank_deficient.append(torch.linalg.pinv(flattened_X[..., :n_eqs, :]) @ flattened_Y[..., :n_eqs, :])
            if len(_rank_deficient) == 0:
                _rank_deficient = torch.zeros_like(_rank_full[..., :0, :, :])
            else:
                _rank_deficient = torch.stack(_rank_deficient, dim=-3)

            # [n_systems x n_traces x (n_positions - 1) x (ir_length * O_D) x O_D]
            # -> [n_systems x n_traces x (n_positions - 1) x ir_length x O_D x O_D]
            # -> [n_systems x n_traces x (n_positions - 1) x O_D x ir_length x O_D]
            observation_IRs = torch.cat([_rank_deficient, _rank_full], dim=-3).unflatten(-2, (ir_length, config.ny)).transpose(dim0=-3, dim1=-2)

            # SECTION: Compute the empirical output
            shifted_X = padded_ys[..., X_indices + 1, :]    # [n_systems x n_traces x (n_positions - 1) x ir_length x O_D]

            # Clean up padded_ys to free memory
            del padded_ys
            torch.cuda.empty_cache()
            gc.collect()

            flattened_observation_IRs = observation_IRs.flatten(0, 2)   # [B x O_D x ir_length x O_D]
            flattened_shifted_X = shifted_X.flatten(0, 2)               # [B x ir_length x O_D]

            # [n_systems x n_traces x (n_positions + 1) x O_D]
            torch_ys = torch.Tensor(ys_sys).to(device)

            del ys_sys
            torch.cuda.empty_cache()
            gc.collect()

            preds_rls_wentinn_sys = torch.vmap(Fn.conv2d)(
                flattened_observation_IRs,                                              # [B x O_D x ir_length x O_D]
                flattened_shifted_X.transpose(dim0=-2, dim1=-1)[..., None, :, :, None]  # [B x 1 x O_D x ir_length x 1]
            ).reshape(*torch_ys.shape[:2], config.n_positions - 1, config.ny) # [n_systems x n_traces x (n_positions - 1)]

            preds_rls_wentinn_sys = torch.cat([
                torch.zeros_like(torch_ys[..., :2, :]),
                preds_rls_wentinn_sys
            ], dim=-2)  # [n_systems x n_traces x (n_positions + 1) x O_D]

            if torch.all(preds_rls_wentinn == 0):
                preds_rls_wentinn = preds_rls_wentinn_sys
            else:
                preds_rls_wentinn = torch.vstack((preds_rls_wentinn, preds_rls_wentinn_sys))

            sim_objs_td = TensorDict({
                "F": torch.Tensor(np.stack([
                    sim_obj.A for sim_obj in sim_objs
                ], axis=0)),
                "H": torch.Tensor(np.stack([
                    sim_obj.C for sim_obj in sim_objs
                ], axis=0)),
                "sqrt_S_W": torch.stack([
                    torch.eye(config.nx) * sim_obj.sigma_w for sim_obj in sim_objs
                ]),
                "sqrt_S_V": torch.stack([
                    torch.eye(config.ny) * sim_obj.sigma_v for sim_obj in sim_objs
                ]),
            }, batch_size=(len(sim_objs),)).to(device)

            # SECTION: Compute analytical errors
            preds_rls_wentinn_analytical_sys = CnnKF.analytical_error(
                observation_IRs,            # [n_systems x n_traces x (n_positions - 1) x ...]
                sim_objs_td[sys, None, None]  # [n_systems x 1 x 1 x ...]
            )   # [n_systems x n_traces x (n_positions - 1)]

            preds_rls_wentinn_analytical_sys = torch.cat([
                torch.norm(torch_ys[..., :2, :], dim=-1) ** 2,    # [n_systems x n_traces x 2]
                preds_rls_wentinn_analytical_sys,               # [n_systems x n_traces x (n_positions - 1)]
            ], dim=-1)  # [n_systems x n_traces x (n_positions + 1)]

            # preds_rls_wentinn_analytical[sys] = preds_rls_wentinn_analytical_sys
            if torch.all(preds_rls_wentinn_analytical == 0):
                preds_rls_wentinn_analytical = preds_rls_wentinn_analytical_sys
            else:
                preds_rls_wentinn_analytical = torch.vstack((preds_rls_wentinn_analytical, preds_rls_wentinn_analytical_sys))

            del torch_ys
            torch.cuda.empty_cache()
            gc.collect()

    return preds_rls_wentinn, preds_rls_wentinn_analytical

def compute_OLS_ir_current(config, ys, sim_objs, max_ir_length, err_lss):
    # set torch precision to float64
    torch.set_default_dtype(torch.float64)
    print("\n\n max_ir_length + 1:", max_ir_length + 1)
    for ir_length in range(1, max_ir_length + 1):
        start = time.time()
        print(f"\n\nIR length: {ir_length}")

        if ir_length == 2:
            preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper_current(config, ys, sim_objs, ir_length, 0.0)

            err_lss[f"OLS_ir_{ir_length}_unreg"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
            err_lss[f"OLS_analytical_ir_{ir_length}_unreg"] = np.array(preds_rls_wentinn_analytical)

        preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper_current(config, ys, sim_objs, ir_length, 1.0)

        err_lss[f"OLS_ir_{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.array(preds_rls_wentinn_analytical)
        end = time.time()
        print("time elapsed:", (end - start) / 60, "min")
    # set torch precision back to float32
    torch.set_default_dtype(torch.float32)
    return err_lss


def compute_OLS_little_helper_current(ls, ls_analytical, sim_obj, padded_ys, ir_length, config, ridge):
    rls_wentinn = CnnKF(config.ny, ir_length, ridge=ridge)
    for i in range(config.n_positions - 1):
        obs_tensor = rls_wentinn.update(
            torch.from_numpy(padded_ys[i:i + ir_length]),
            torch.from_numpy(padded_ys[i + ir_length])
        )

        ls.append(
            rls_wentinn(torch.from_numpy(padded_ys[i + 1:i + ir_length + 1])[None]).squeeze(0, 1).detach().numpy())
        ls_analytical.append(rls_wentinn.analytical_error(sim_obj).item())

        # assert ls_analytical[-1] >= torch.trace(sim_obj.S_observation_inf).item(), f"Analytical error is less than irreducible error: {ls_analytical[-1]} < {torch.trace(sim_obj.S_observation_inf).item()}."
    return ls, ls_analytical


def compute_OLS_helper_current(config, ys, sim_objs, ir_length, ridge):
    preds_rls_wentinn = []
    preds_rls_wentinn_analytical = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _preds_rls_wentinn = []
        _preds_rls_wentinn_analytical = []
        for __ys in _ys:
            padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])  # [(L + R - 1) x O_D]
            ls = list(np.zeros((2, config.ny)))
            ls_analytical = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)

            ls, ls_analytical = compute_OLS_little_helper_current(ls, ls_analytical, sim_obj, padded_ys, ir_length, config,
                                                          ridge)

            _preds_rls_wentinn.append(ls)
            _preds_rls_wentinn_analytical.append(ls_analytical)

        preds_rls_wentinn.append(_preds_rls_wentinn)
        preds_rls_wentinn_analytical.append(_preds_rls_wentinn_analytical)
    return preds_rls_wentinn, preds_rls_wentinn_analytical


def compute_OLS_wentinn(config, ys, sim_objs, ir_length, err_lss):
    errs_rls_wentinn = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _errs_rls_wentinn = []
        for __ys in _ys:
            padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])  # [(L + R - 1) x O_D]
            ls = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)
            rls_wentinn = CnnKF(config.ny, ir_length)
            for i in range(config.n_positions - 1):
                rls_wentinn.update(
                    torch.from_numpy(padded_ys[i:i + ir_length]),
                    torch.from_numpy(padded_ys[i + ir_length])
                )
                ls.append(rls_wentinn.analytical_error(sim_obj).item())
            _errs_rls_wentinn.append(ls)
        errs_rls_wentinn.append(_errs_rls_wentinn)
    err_lss["OLS_wentinn"] = np.array(errs_rls_wentinn)
    return err_lss


def batch_trace(x):
    # Ensure x has at least two dimensions
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least two dimensions")
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def compute_errors(config, C_dist, run_deg_kf_test, wentinn_data):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger

    print("val_dataset_typ:", config.val_dataset_typ)
    num_systems = config.num_val_tasks  # number of validation tasks
    print("Number of validation systems:", num_systems)
    num_trials = config.num_traces["val"]  # number of traces
    print("Number of traces:", num_trials)

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head).eval().to(
        device)  # load_from_checkpoint

    if wentinn_data:

        with open(f"../data/numpy_three_sys" + C_dist + "/test_sim.pt", "rb") as f:
            sim_objs = torch.load(f)

        with open('../data/numpy_three_sys' + C_dist + '/data.pkl',
                  'rb') as f:  # load the data.pkl file for the test data
            data = pickle.load(f)
            ys = data["observation"]
            print("ys.shape:", ys.shape)
    else:
        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)

        print("getting the validation data")
        # open fsim file
        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}.pkl", "rb") as f:
            samples = pickle.load(f)
            # for every 2000 entries in samples, get the observation values and append them to the ys list
            ys = np.stack(
                [entry["obs"] for entry in samples], axis=0
            ).reshape((num_systems, num_trials, config.n_positions + 1, config.ny)).astype(np.float32)

            prev_xs = np.concatenate([
                np.zeros((num_systems, num_trials, 1, config.nx)),
                np.stack(
                    [entry["states"][:-1] for entry in samples], axis=0
                ).reshape((num_systems, num_trials, config.n_positions, config.nx)).astype(np.float32)
            ], axis=2)
            noiseless_ys = prev_xs @ np.stack([sim_obj.C @ sim_obj.A for sim_obj in sim_objs], axis=0)[:, None].transpose(0, 1, 3, 2)

            gc.collect()  # Start the garbage collector

    # print("no tf pred")
    # Transformer Predictions
    print("start tf pred")
    start = time.time()  # start the timer for transformer predictions
    with torch.no_grad():  # no gradients
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)
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
            validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I),
                                                            batch_size=config.test_batch_size)
            preds_arr = []  # Store the predictions for all batches
            for validation_batch in iter(validation_loader):
                _, flattened_preds_tf = model.predict_step(
                    {"xs": validation_batch.to(device)})  # .float().to(device)})    # predict using the model
                preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
            preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                                  (*batch_shape, *I.shape[-2:]))  # Combine the predictions for all batches
            # print("preds_tf.shape:", preds_tf.shape)
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                      axis=-2)  # concatenate the predictions
            # print("preds_tf.shape:", preds_tf.shape)
    end = time.time()  # end the timer for transformer predictions
    print("time elapsed for MOP Pred:", (end - start) / 60, "min")  # print the time elapsed for transformer predictions

    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
    noiseless_errs_tf = np.linalg.norm((noiseless_ys - preds_tf), axis=-1) ** 2 + np.array([
        (np.linalg.norm(sim_obj.C) * sim_obj.sigma_w) ** 2 + config.ny * (sim_obj.sigma_v ** 2)
        for sim_obj in sim_objs
    ])[:, None, None]

    print("zero predictor")
    # zero predictor predictions
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2  # get the errors of zero predictions

    n_noise = config.n_noise

    start = time.time()  # start the timer for kalman filter predictions
    if run_deg_kf_test:  # degenerate system KF Predictions

        #############################################################
        # this portion can most likely be deleted
        # Kalman Filter Predictions
        preds_kf_list = []
        for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)):
            inner_list = []
            for __ys in _ys:
                result = apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise),
                                  sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
                inner_list.append(result)
            preds_kf_list.append(inner_list)
        #############################################################

        preds_kf = np.array(preds_kf_list)  # get kalman filter predictions

        # create an array of zeros to hold the kalman filter predictions
        preds_kf = np.zeros((num_systems, num_systems, num_trials, config.n_positions + 1,
                             config.ny))  # first axis is the system that the kalman filter is being trained on, second axis is the system that the kalman filter is being tested on

        errs_kf = np.zeros((num_systems, num_systems, num_trials,
                            config.n_positions + 1))  # first axis is the system that the kalman filter is being trained on, second axis is the system that the kalman filter is being tested on
        # iterate over sim_objs
        kf_index = 0
        for sim_obj in sim_objs:  # iterate over the training systems
            for sys in range(num_systems):  # iterate over the test systems
                print("Kalman filter", kf_index, "testing on system", sys)
                for trial in range(num_trials):
                    preds_kf[kf_index, sys, trial, :, :] = apply_kf(sim_obj, ys[sys, trial, :-1, :],
                                                                    sigma_w=sim_obj.sigma_w * np.sqrt(n_noise),
                                                                    sigma_v=sim_obj.sigma_v * np.sqrt(
                                                                        n_noise))  # get the kalman filter predictions for the test system and the training system
                errs_kf[kf_index, sys] = np.linalg.norm((ys[sys] - preds_kf[kf_index, sys]), axis=-1) ** 2  # get the errors of the kalman filter predictions for the test system and the training system
            kf_index += 1

    else:  
        # print("no kf pred")
        # Kalman Predictions
        print("start kf pred")
        preds_kf = np.array([[
            apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise),
                     sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
            for __ys in _ys
        ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
        ])  # get kalman filter predictions
        errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2

    end = time.time()  # end the timer for kalman filter predictions
    print("time elapsed for KF Pred:", (end - start) / 60,
          "min")  # print the time elapsed for kalman filter predictions

    err_lss = collections.OrderedDict([
        ("Kalman", errs_kf),
        ("MOP", errs_tf),
        ("MOP_analytical", noiseless_errs_tf),
        ("Zero", errs_zero)
    ])
    del preds_kf
    del errs_kf
    del preds_tf
    del errs_tf
    del noiseless_errs_tf
    del noiseless_ys
    del errs_zero

    torch.cuda.empty_cache()
    gc.collect()

    # # Check if CUDA is available
    # if torch.cuda.is_available():

    #     # Print memory usage
    #     print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
    #     print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
    #     print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")
    #     print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 2):.2f} MB")
    # else:
    #     print("CUDA is not available.")


    # Analytical Kalman Predictions
    analytical_kf = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    err_lss["Analytical_Kalman"] = analytical_kf.reshape((num_systems, 1)) @ np.ones((1, config.n_positions))

    #Analytical simulation predictions
    #generate config.n_positions multivariate normal random variables with mean zero and covariance sim_obj.S_observation_inf and do this config.num_traces["val"] times for each sim_obj
    an_sims = np.array([np.random.multivariate_normal(np.zeros(config.ny), sim_obj.S_observation_inf, (config.num_traces["val"], config.n_positions+1)) for sim_obj in sim_objs])

    print("an_sims shape:", an_sims.shape)
    err_lss["Analytical_Simulation"] = np.linalg.norm(an_sims, axis=-1) ** 2

    # Original OLS
    # Clear the PyTorch cache
    start = time.time()  # start the timer for OLS predictions
    print("start OLS pred")
    #print(torch.cuda.memory_summary())
    err_lss = compute_OLS_ir(config, ys, sim_objs, max_ir_length=3, err_lss=err_lss)
    end = time.time()  # end the timer for OLS predictions
    print("time elapsed for OLS Pred:", (end - start) / 60, "min")  # print the time elapsed for OLS predictions

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    return err_lss, irreducible_error


def compute_errors_conv(config, C_dist, run_deg_kf_test, wentinn_data):
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

    if wentinn_data:

        with open(f"../data/numpy_three_sys" + C_dist + "/test_sim.pt", "rb") as f:
            sim_objs = torch.load(f)

        with open('../data/numpy_three_sys' + C_dist + '/data.pkl',
                  'rb') as f:  # load the data.pkl file for the test data
            data = pickle.load(f)
            ys = data["observation"]
            print("ys.shape:", ys.shape)
    else:
        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)


        # open fsim file
        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}.pkl", "rb") as f:
            samples = pickle.load(f)
            # for every 2000 entries in samples, get the observation values and append them to the ys list
            ys = np.stack(
                [entry["obs"] for entry in samples], axis=0
            ).reshape((num_systems, num_trials, config.n_positions + 1, config.ny)).astype(np.float32)

            prev_xs = np.concatenate([
                np.zeros((num_systems, num_trials, 1, config.nx)),
                np.stack(
                    [entry["states"][:-1] for entry in samples], axis=0
                ).reshape((num_systems, num_trials, config.n_positions, config.nx)).astype(np.float32)
            ], axis=2)

            # Debugging: Print the shape of each matrix multiplication result
            for idx, sim_obj in enumerate(sim_objs):
                result = sim_obj.C @ sim_obj.A

            # Stack the results
            stacked_matrix = np.stack([sim_obj.C @ sim_obj.A for sim_obj in sim_objs], axis=0)

            reshaped_matrix = stacked_matrix[:, None].transpose(0,1,3,2)

            noiseless_ys = prev_xs @ reshaped_matrix

            gc.collect()  # Start the garbage collector

    # Transformer Predictions
    start = time.time()  # start the timer for transformer predictions
    print("started transformer predictions")
    with torch.no_grad():  # no gradients
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)
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
            validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I),
                                                            batch_size=config.test_batch_size)
            preds_arr = []  # Store the predictions for all batches
            for validation_batch in iter(validation_loader):
                _, flattened_preds_tf = model.predict_step(
                    {"xs": validation_batch.to(device)})  # .float().to(device)})    # predict using the model
                preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
            preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                                  (*batch_shape, *I.shape[-2:]))  # Combine the predictions for all batches
            # print("preds_tf.shape:", preds_tf.shape)
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                      axis=-2)  # concatenate the predictions
            # print("preds_tf.shape:", preds_tf.shape)
    end = time.time()  # end the timer for transformer predictions
    print("time elapsed for MOP Pred:", (end - start) / 60, "min")  # print the time elapsed for transformer predictions

    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
    noiseless_errs_tf = np.linalg.norm((noiseless_ys - preds_tf), axis=-1) ** 2 + np.array([
        (np.linalg.norm(sim_obj.C) * sim_obj.sigma_w) ** 2 + config.ny * (sim_obj.sigma_v ** 2)
        for sim_obj in sim_objs])[:, None, None]

    # zero predictor predictions
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2  # get the errors of zero predictions

    err_lss = collections.OrderedDict([
        ("MOP", errs_tf),
        ("MOP_analytical", noiseless_errs_tf),
        ("Zero", errs_zero)
    ])
    print("err_lss keys:", err_lss.keys())

    print("len of sim_objs:", len(sim_objs))
    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    print("len of irreducible_error:", len(irreducible_error))
    return err_lss, irreducible_error


def save_preds(run_deg_kf_test, config):
    err_lss, irreducible_error = compute_errors(config, config.C_dist, run_deg_kf_test,
                                                wentinn_data=False)  # , emb_dim)

    # make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    # get the step size from the ckpt_path
    step_size = config.ckpt_path.split("/")[-1].split("_")[-1]  # get step number

    os.makedirs(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size, exist_ok=True)
    if run_deg_kf_test:
        # save err_lss and irreducible_error to a file
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + f"/{config.val_dataset_typ}_err_lss_deg_kf_test.pkl",
                "wb") as f:
            pickle.dump(err_lss, f)
    else:
        # save err_lss and irreducible_error to a file
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.val_dataset_typ}_err_lss.pkl",
                "wb") as f:
            pickle.dump(err_lss, f)
            print("err_lss keys", err_lss.keys())

    with open(
            parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.val_dataset_typ}_irreducible_error.pkl",
            "wb") as f:
        pickle.dump(irreducible_error, f)

def save_preds_conv_helper(save_dir, run_deg_kf_test, config):
    os.makedirs(save_dir, exist_ok=True)

    err_lss, irreducible_error = compute_errors_conv(config, config.C_dist, run_deg_kf_test,
                                                        wentinn_data=False)  # , emb_dim)

    print("helper len of irreducible_error:", len(irreducible_error))
    # save err_lss and irreducible_error to a file
    with open(
            save_dir + f"/{config.val_dataset_typ}_err_lss.pkl",
            "wb") as f:
        pickle.dump(err_lss, f)

    with open(
            save_dir + f"/{config.val_dataset_typ}_irreducible_error.pkl",
            "wb") as f:
        pickle.dump(irreducible_error, f)
    return


def save_preds_conv(make_preds, run_deg_kf_test, config):
    # get the step size from the ckpt_path
    step_size = config.ckpt_path.split("/")[-1].split("_")[-1]
    print("step_size: %r" % step_size)

    # make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    save_dir = parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size

    # a boolean for whether the below directory exists
    #check if a specific file named config.val_dataset_typ_err_lss.pkl exists in the directory
    if os.path.exists(save_dir + f"/{config.val_dataset_typ}_err_lss.pkl"):
        print(f"{config.val_dataset_typ}_err_lss.pkl for ", step_size, " already exists")
    else:
        print(f"{config.val_dataset_typ}_err_lss.pkl for ", step_size, " does not exist")
        save_preds_conv_helper(save_dir, run_deg_kf_test, config)
    return

        


def load_preds(run_deg_kf_test, excess, num_systems, config):
    # make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    # get the step size from the ckpt_path
    step_size = config.ckpt_path.split("/")[-1].split("_")[-1]
    print("step_size:", step_size)

    if run_deg_kf_test:
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + f"/{config.val_dataset_typ}_err_lss_deg_kf_test.pkl",
                "rb") as f:
            err_lss_load = pickle.load(f)
            print("len(err_lss_load):", len(err_lss_load))
    else:
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.val_dataset_typ}_err_lss.pkl",
                "rb") as f:
            err_lss_load = pickle.load(f)

    print("err_lss_load keys:", err_lss_load.keys())

    with open(
            parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.val_dataset_typ}_irreducible_error.pkl",
            "rb") as f:
        irreducible_error_load = pickle.load(f)

    if config.C_dist == "_unif_C" and config.val_dataset_typ == "ypred":
        with open(f"../data/prediction_errors_unif_C/fir_bounds.pt", "rb") as f:
            fir_bounds = torch.load(f, map_location=torch.device('cpu'))
            fir_bounds = fir_bounds.T

        # with open(f"../data/prediction_errors_unif_C/wentinn_errors.pt", "rb") as f:
        #     rnn_errors = torch.load(f, map_location=torch.device('cpu'))

        # with open(f"../data/prediction_errors_unif_C/rnn_analytical_errors.pt", "rb") as f:
        #     rnn_an_errors = torch.load(f, map_location=torch.device('cpu'))
        #     rnn_an_errors = rnn_an_errors.permute(1,2,0)

        with open(f"../data/wentinn_12_04_24/errors.pt", "rb") as f:
            rnn_errors = torch.load(f, map_location=torch.device('cpu'))
            rnn_errors = rnn_errors.permute(1, 2, 0)

        with open(f"../data/wentinn_12_04_24/analytical_errors.pt", "rb") as f:
            rnn_an_errors = torch.load(f, map_location=torch.device('cpu'))
            rnn_an_errors = rnn_an_errors.permute(1, 2, 0)
    else:
        fir_bounds = np.zeros((num_systems, 1))
        rnn_errors = np.zeros((num_systems, 1, 32))
        rnn_an_errors = np.zeros((num_systems, 1, 32))

    return err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors


def setup_deg_kf_axs_arrs(num_systems):
    # create a square array of zeros that is the size of the number of systems to hold the cosine similarities
    cos_sims = np.zeros((num_systems, num_systems))
    err_ratios = np.zeros((num_systems, num_systems))
    zero_ratios = np.zeros((num_systems, num_systems))

    deg_fig = plt.figure(figsize=(40, 40))
    ax1 = deg_fig.add_subplot(331)  # This creates the first subplot
    ax2 = deg_fig.add_subplot(332)  # This creates the second subplot
    ax3 = deg_fig.add_subplot(333)  # This creates the third subplot
    ax4 = deg_fig.add_subplot(334)  # This creates the third subplot
    ax5 = deg_fig.add_subplot(335)  # This creates the third subplot
    ax6 = deg_fig.add_subplot(336)  # This creates the third subplot
    ax7 = deg_fig.add_subplot(337)  # This creates the third subplot
    ax8 = deg_fig.add_subplot(338)  # This creates the third subplot
    ax9 = deg_fig.add_subplot(339)  # This creates the third subplot
    axs = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]]
    return cos_sims, err_ratios, zero_ratios, deg_fig, axs


def create_plots(config, run_preds, run_deg_kf_test, excess, num_systems, shade, logscale):
    C_dist = config.C_dist
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)

    if run_preds:
        print("config path:", config.ckpt_path)
        save_preds(run_deg_kf_test, config)  # save the predictions to a file

    # load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess,
                                                                                             num_systems, config)

    if run_deg_kf_test:
        cos_sims, err_ratios, zero_ratios, deg_fig, axs = setup_deg_kf_axs_arrs(num_systems)

    # colors = [ '#EE7733', '#0077BB', '#EE3377', '#CC3311', '#009988', '#DDDDDD', '#33BBEE', '#EEDD88', '#BBBBBB','#7D00BD', '#d00960', '#006400', '#ff1493', '#00ff00', '#ff4500', '#8a2be2', '#5f9ea0', '#d2691e','#ff6347', '#4682b4', '#daa520', '#7fff00']

   # Define the dark colors in hex format
    colors = [
    '#1f77b4',  # Dark Blue
    '#ff7f0e',  # Dark Orange
    '#2ca02c',  # Dark Green
    '#d62728',  # Dark Red
    '#9467bd',  # Purple
    '#17becf',  # Dark Cyan
    '#8c564b',  # Dark Brown
]

    print("len(err_lss_load):", len(err_lss_load))
    for sys in range(len(irreducible_error_load)):

        if run_deg_kf_test:
            for i in range(num_systems):
                err_lss_copy = copy.deepcopy(err_lss_load)
                err_lss_copy["Kalman"] = err_lss_copy["Kalman"][i]

                print("KF trained on system", i, "testing on system", sys)

                # plot transformer, KF and FIR errors
                handles, err_rat = plot_errs(colors, sys, err_lss_copy, irreducible_error_load, ax=axs[i][sys],
                                             shade=True, normalized=logscale)

                err_ratios[i, sys] = err_rat[0]
                zero_ratios[i, sys] = err_rat[1]

                # compute the cosine similarity between the err_lss_load["Kalman"][i][sys] and err_lss_load[i][i]
                cos_sim = np.dot(err_lss_load["Kalman"][i][sys].flatten(), err_lss_load["Kalman"][i][i].flatten()) / (
                            np.linalg.norm(err_lss_load["Kalman"][i][sys]) * np.linalg.norm(
                        err_lss_load["Kalman"][sys][sys]))
                print("cosine similarity between KF trained on system", i, "testing on system", sys,
                      "and KF trained and tested on system", sys, ":", cos_sim)
                cos_sims[i, sys] = cos_sim

                if C_dist == "_unif_C" and config.val_dataset_typ == "ypred":
                    # plot fir bounds
                    for j in range(fir_bounds.shape[1] - 2):
                        handles.extend(axs[i][sys].plot(np.array(range(config.n_positions)),
                                                        fir_bounds[sys, j] * np.ones(config.n_positions),
                                                        label="IR Analytical Length " + str(j + 1), linewidth=3,
                                                        linestyle='--', color=colors[j + 5]))

                    # plot RNN errors
                    avg, std = rnn_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_errors.shape[1])) * rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(
                        axs[i][sys].scatter(np.arange(0, 32 * 5, 5), avg_numpy, label="RNN", linewidth=1, marker='x',
                                            s=50, color=colors[len(err_lss_copy)]))
                    axs[i][sys].fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy,
                                             avg_numpy + std_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_an_errors.shape[1])) * rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(
                        axs[i][sys].scatter(np.arange(0, 251, 5), avg_an_numpy, label="RNN Analytical", linewidth=1,
                                            marker='o', s=100, color=colors[len(err_lss_copy)], zorder=10))
                    axs[i][sys].fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy,
                                             avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0],
                                             alpha=0.2)

                del err_lss_copy  # Delete the variable
                gc.collect()  # Start the garbage collector

                axs[i][sys].legend(fontsize=18, loc="upper right", ncol=math.floor(len(handles) / 4))
                axs[i][sys].set_xlabel("i", fontsize=30)
                axs[i][sys].set_ylabel("Prediction Error", fontsize=30)
                axs[i][sys].grid(which="both")
                axs[i][sys].tick_params(axis='both', which='major', labelsize=30)
                axs[i][sys].tick_params(axis='both', which='minor', labelsize=20)
                axs[i][sys].set_title("KF system " + str(i) + " testing on system " + str(sys) + (
                    ": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                        ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else ": Dense A ")) + (
                                          "Uniform C" if C_dist == "_unif_C" else (
                                              "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                axs[i][sys].set_ylim(bottom=10 ** (-0.7), top=2 * 10 ** (0))
                # axs[i][sys].set_xlim(left=0, right=10)

            if sys == num_systems - 1 and i == num_systems - 1:
                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                deg_fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + (
                    "-changing_deg_kf_test" if config.changing else "deg_kf_test"))
        else:
            fig = plt.figure(figsize=(15, 15)) # create a figure with a size of 15x15
            ax = fig.add_subplot(111)
            # plot transformer, KF and FIR errors
            handles, err_rat = plot_errs(colors, sys, err_lss_load, irreducible_error_load, ax=ax, shade=shade,
                                         normalized=logscale)

            # if C_dist == "_unif_C" and config.dataset_typ == "ypred":
            #     if excess:
            #         # plot fir bounds
            #         for i in range(fir_bounds.shape[1] - 2):
            #             handles.extend(ax.plot(np.array(range(config.n_positions)),
            #                                    (fir_bounds[sys, i] - irreducible_error_load[sys]) * np.ones(
            #                                        config.n_positions),
            #                                    label="IR Analytical Length " + str(i + 1) + " sys: " + str(sys),
            #                                    linewidth=3, linestyle='--'))  # , color = colors[i + 5]))

            #         # #plot RNN errors
            #         # rnn_er = rnn_errors[sys].detach().numpy()
            #         # kalman_err = err_lss_load["Kalman"][sys,:, ::5].mean(axis=(0))
            #         # #figure out how to take median and quantiles of the rnn errors
            #         # rnn_q1, rnn_median, rnn_q3 = np.quantile((rnn_er -kalman_err), [0.25, 0.5, 0.75], axis=-2)
            #         # scale = rnn_median[1]
            #         # rnn_median = rnn_median/scale
            #         # rnn_q1 = rnn_q1/scale
            #         # rnn_q3 = rnn_q3/scale
            #         # N = rnn_median.shape[0]
            #         # # Adjust the range of np.arange function
            #         # x = np.arange(1, (N-1)*5 + 1, 5)
            #         # handles.append(ax.scatter(x, rnn_median[1:], label="RNN sys: " + str(sys), linewidth=3, marker='x', s=50))#, color=colors[len(err_lss_load)]))
            #         # if shade:
            #         #     ax.fill_between(x, rnn_q1[1:], rnn_q3[1:], facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

            #         print("rnn_an_errors.shape:", rnn_an_errors.shape)
            #         # plot Analytical RNN errors
            #         rnn_an_er = rnn_an_errors[sys].detach().numpy()
            #         print("shape of err_lss_load[Kalman]:", err_lss_load["Kalman"][sys, :, ::5].shape)
            #         kalman_err = err_lss_load["Kalman"][sys, :, ::5].mean(axis=(0))
            #         # figure out how to take median and quantiles of the rnn errors
            #         rnn_an_q1, rnn_an_median, rnn_an_q3 = np.quantile((rnn_an_er - kalman_err), [0.25, 0.5, 0.75],
            #                                                           axis=-2)
            #         scale = rnn_an_median[1]
            #         rnn_an_median = rnn_an_median / scale
            #         rnn_an_q1 = rnn_an_q1 / scale
            #         rnn_an_q3 = rnn_an_q3 / scale
            #         N = rnn_an_median.shape[0]
            #         # Adjust the range of np.arange function
            #         x = np.arange(1, (N - 1) * 5 + 1, 5)
            #         handles.append(
            #             ax.scatter(x, rnn_an_median[1:], label="RNN Analytical sys: " + str(sys), linewidth=1,
            #                        marker='o', s=100))  # , color=colors[len(err_lss_load)]))
            #         if shade:
            #             ax.fill_between(x, rnn_an_q1[1:], rnn_an_q3[1:], facecolor=handles[-1].get_facecolor()[0],
            #                             alpha=0.2)
            #         # avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
            #         # avg_an_numpy = avg_an.detach().numpy()
            #         # std_an_numpy = std_an.detach().numpy()
            #         # handles.append(ax.scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100))#, color=colors[len(err_lss_load)], zorder=10))
            #         # ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)
            #     else:
            #         # plot fir bounds
            #         for i in range(fir_bounds.shape[1] - 2):
            #             handles.extend(ax.plot(np.array(range(config.n_positions)),
            #                                    fir_bounds[sys, i] * np.ones(config.n_positions),
            #                                    label="IR Analytical Length " + str(i + 1), linewidth=3, linestyle='--',
            #                                    color=colors[i + 5]))

            #         # plot RNN errors
            #         print("rnn_errors.shape:", rnn_errors.shape)
            #         avg, std = rnn_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
            #             rnn_errors.shape[1])) * rnn_errors.std(axis=(0, 1))
            #         avg_numpy = avg.detach().numpy()
            #         std_numpy = std.detach().numpy()
            #         handles.append(
            #             ax.scatter(np.arange(0, config.n_positions + 1, 5), avg_numpy, label="RNN", linewidth=1,
            #                        marker='x', s=50, color=colors[len(err_lss_load)]))
            #         ax.fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy, avg_numpy + std_numpy,
            #                         facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

            #         avg_an, std_an = rnn_an_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
            #             rnn_an_errors.shape[1])) * rnn_an_errors.std(axis=(0, 1))
            #         avg_an_numpy = avg_an.detach().numpy()
            #         std_an_numpy = std_an.detach().numpy()
            #         handles.append(
            #             ax.scatter(np.arange(0, 251, 5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o',
            #                        s=100, color=colors[len(err_lss_load)], zorder=10))
            #         ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy,
            #                         avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

            if excess:
                ncol = 1 if len(handles) < 4 else math.floor(len(handles) / 4)
                ax.legend(fontsize=18, loc="lower left", ncol=ncol)
                ax.set_xlabel("log(t)", fontsize=30)
                ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                # make the x axis log scale
                ax.set_xscale('log')
                # ax.set_ylim(bottom=-1, top=2*10**(-1))
                ax.set_title("System " + str(sys) +(": Rotated Diagonal (Uniform Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal (Gaussian Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))))
                # ax.set_xlim(left=0, right=10)

                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(
                    parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + (
                        "-changing" if config.changing else "_excess"))
            else:
                ax.legend(fontsize=16, loc="upper right", ncol=max(1, math.floor(len(handles) / 2)))
                ax.set_xlabel("i", fontsize=30)

                ax.set_ylabel("Median of Err / Empirical KF Err" if logscale else "Prediction Error", fontsize=30)
                # ax.set_ylabel("Err - Empirical KF Err" if logscale else "Prediction Error", fontsize=30)
                # ax.set_ylabel("Median Error" if logscale else "Avg Error", fontsize=30)

                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                if logscale:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                # ax.set_yscale('linear')
                # ax.set_xscale('linear')


                # if not logscale:
                #     ax.set_ylim(bottom=10 ** (-0.7), top=0.5 * 10 ** (1))  # set the y axis limits

                ax.set_title("System " + str(sys) + (": Rotated Diagonal (|N(0,1)| <= 0.95 Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal (Unif(-1,1) Eigs) A " if config.val_dataset_typ == "rotDiagA_unif" else (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))))
                # ax.set_xlim(left=0, right=10)

                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                #add the date and time to the filename
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")

                #get only the ckpt step from the ckpt_path
                ckpt_step = config.ckpt_path.split("=")[1].split(".")[0]

                #add a caption to the bottom of the figure
                fig.text(0.5, 0.01, "step=" + ckpt_step + "_" + timestamp, ha='center', fontsize=30)
                fig.savefig(
                    parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + (
                        "logscale" if logscale else "") + "_" + "step=" + ckpt_step + "_" + timestamp) 

    if run_deg_kf_test:
        # Create a DataFrame from the numpy array
        # create a list of strings that correspond to the system numbers
        test_col = ["Test sys " + str(i) for i in range(num_systems)]
        train_col = ["Train sys " + str(i) for i in range(num_systems)]

        df = pd.DataFrame(cos_sims, columns=test_col, index=train_col)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Cosine Similarities of KF Predictions')

        # Create a table and save it as an image
        tbl = table(ax, df, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_cosine_similarities_deg_kf_test")

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of KF Predictions')
        # Create a table and save it as an image
        df2 = pd.DataFrame(err_ratios, columns=test_col, index=train_col)
        tbl = table(ax, df2, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_error_ratios_deg_kf_test")

        print("zero_ratios:", zero_ratios)
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of Zero Predictions')
        # Create a table and save it as an image
        df3 = pd.DataFrame(zero_ratios[0, :].reshape(1, -1), columns=test_col)
        tbl = table(ax, df3, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_zero_ratios_deg_kf_test")

    if excess:
        ncol = 1 if len(handles) < 4 else math.floor(len(handles) / 2)
        ax.legend(fontsize=14, loc="lower left", ncol=ncol)
        ax.set_xlabel("log(t)", fontsize=30)
        ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
        ax.grid(which="both")
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        # make the x axis log scale
        ax.set_xscale('log')
        # ax.set_ylim(bottom=-1, top=2*10**(-1))
        ax.set_title(("Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
            "Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                "N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ("Uniform A" if config.val_dataset_typ == "unifA" else "Dense A ")))) + (
                         "Uniform C" if C_dist == "_unif_C" else (
                             "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
        # ax.set_xlim(left=0, right=10)
        os.makedirs(parent_parent_dir + f"/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff" + (
            "-changing" if config.changing else "_excess"))

    return None

# Sort handles and labels based on "MOP" part or "_analytical" part
# Extracting the number after "MOP" or "_analytical" and using it for sorting
def extract_sort_key(label):
    if "_analytical" in label:
        return int(label.split("_analytical")[1])
    else:
        return int(label.split("MOP")[1])


def convergence_plots(j, config, run_preds, run_deg_kf_test, kfnorm, num_systems, shade, fig, ax, ts, kal_errors):
    excess = False
    C_dist = config.C_dist
    print("\n\n", "config path:", config.ckpt_path)
    if run_preds:
        print("\n\nRunning predictions")
        save_preds_conv(run_preds, run_deg_kf_test, config)  # save the predictions to a file
    print("\n\nLoading predictions")
    # load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess,
                                                                                             num_systems, config)

    colors = [
        '#D32F2F',  # Red
        '#C2185B',  # Pink
        '#7B1FA2',  # Purple
        '#512DA8',  # Deep Purple
        '#303F9F',  # Indigo
        '#1976D2',  # Blue
        '#0288D1',  # Light Blue
        '#0097A7',  # Cyan
        '#00796B',  # Teal
        '#388E3C',  # Green
        '#689F38',  # Light Green
        '#AFB42B',  # Lime
        '#FBC02D',  # Yellow
        '#FFA000',  # Amber
        '#F57C00',  # Orange
        '#E64A19',  # Deep Orange
        '#5D4037',  # Brown
        '#616161',  # Grey
        '#455A64',  # Blue Grey
        '#8E24AA',  # Purple 600
        '#D81B60',  # Pink 600
        '#3949AB',  # Indigo 600
        '#F4511E',  # Deep Orange 600
        '#6D4C41',  # Brown 600
        '#1B5E20',  # Dark Green
        '#33691E',  # Lime Green Dark
        '#827717',  # Olive
        '#F9A825',  # Mustard
        '#FF6F00',  # Orange Deep
        '#E65100',  # Orange Dark
        '#BF360C',  # Deep Orange Dark
        '#3E2723',  # Deep Brown
        '#212121',  # Almost Black
        '#263238',  # Blue Grey Dark
        '#004D40',  # Teal Dark
        '#006064',  # Cyan Dark
        '#01579B',  # Light Blue Dark
        '#0D47A1',  # Blue Dark
        '#1A237E',  # Indigo Dark
        '#311B92',  # Deep Purple Dark
        '#4A148C',  # Purple Dark
        '#880E4F',  # Pink Dark
        '#B71C1C',  # Red Dark
        '#D50000',  # Red Accent
        '#C51162',  # Pink Accent
        '#AA00FF',  # Purple Accent
        '#6200EA',  # Deep Purple Accent
        '#304FFE',  # Indigo Accent
    ]
    print("\n\nPlotting predictions")
    sys_errs = []
    sys_errs_an = []
    for sys in range(len(irreducible_error_load)):
        # plot transformer, KF and FIR errors
        # get the checkpoint steps number from the checkpoint path
        ckpt_steps = config.ckpt_path.split("step=")[1].split(".")[0]  # get the checkpoint steps number
        handles, err_avg_t, err_avg_t_an = plot_errs_conv(ts, j, colors, sys, err_lss_load, irreducible_error_load, ckpt_steps,
                                            kfnorm, ax=ax[sys], shade=shade, kal_err=kal_errors)  # plot the errors
        sys_errs.append(err_avg_t)  # append the system number and the error average at step t
        sys_errs_an.append(err_avg_t_an)  # append the system number and the error average at step t

        # Step 1: Collect legend handles and labels
        handles, labels = ax[sys].get_legend_handles_labels()  # handles and labels of the legend

        # Step 2: Sort handles and labels
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda hl: extract_sort_key(hl[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)

        # Step 3: Create the legend with sorted handles and labels
        ax[sys].legend(sorted_handles, sorted_labels, fontsize=18, loc="upper right", ncol=1)

        # ax[sys].legend(fontsize=18, loc="upper right", ncol=1)
        ax[sys].set_xlabel("t", fontsize=30)
        ax[sys].set_ylabel("Prediction Error", fontsize=30)
        ax[sys].grid(which="both")
        ax[sys].tick_params(axis='both', which='major', labelsize=30)
        ax[sys].tick_params(axis='both', which='minor', labelsize=20)

        # set y axis limits
        ax[sys].set_ylim(bottom=10 ** (-2), top=5 * 10 ** (1))

        ax[sys].set_title("System " + str(sys) + (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
            ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                              "Uniform C" if C_dist == "_unif_C" else (
                                  "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")) + (
                              " Normalized" if kfnorm else ""), fontsize=20)
        # ax.set_xlim(left=0, right=10)

    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
    fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_conv" + (
        "_normalized" if kfnorm else "") + ("-changing" if config.changing else ""))

    return (ckpt_steps, sys_errs), (ckpt_steps, sys_errs_an)  # return the checkpoint steps number and the system errors


####################################################################################################
# main function
if __name__ == '__main__':
    config = Config()

    C_dist = "_gauss_C"  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var"
    run_preds = True  # run the predictions evaluation
    run_deg_kf_test = False  # run degenerate KF test
    excess = False  # run the excess plots
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)
    shade = False

    num_systems = config.num_val_tasks  # number of validation tasks

    if run_preds:
        print("config path:", config.ckpt_path)
        save_preds(run_deg_kf_test, config)  # save the predictions to a file

    # load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess,
                                                                                             num_systems, config)

    if run_deg_kf_test:
        cos_sims, err_ratios, zero_ratios, deg_fig, axs = setup_deg_kf_axs_arrs(num_systems)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#A80000', '#bcbd22']

    print("len(err_lss_load):", len(err_lss_load))
    for sys in range(len(irreducible_error_load)):

        if run_deg_kf_test:
            for i in range(num_systems):
                err_lss_copy = copy.deepcopy(err_lss_load)
                err_lss_copy["Kalman"] = err_lss_copy["Kalman"][i]

                print("KF trained on system", i, "testing on system", sys)

                # plot transformer, KF and FIR errors
                handles, err_rat = plot_errs(colors, sys, err_lss_copy, irreducible_error_load, ax=axs[i][sys],
                                             shade=True, normalized=excess)
                print("err_rat:", err_rat)

                err_ratios[i, sys] = err_rat[0]
                zero_ratios[i, sys] = err_rat[1]

                # compute the cosine similarity between the err_lss_load["Kalman"][i][sys] and err_lss_load[i][i]
                cos_sim = np.dot(err_lss_load["Kalman"][i][sys].flatten(), err_lss_load["Kalman"][i][i].flatten()) / (
                            np.linalg.norm(err_lss_load["Kalman"][i][sys]) * np.linalg.norm(
                        err_lss_load["Kalman"][sys][sys]))
                print("cosine similarity between KF trained on system", i, "testing on system", sys,
                      "and KF trained and tested on system", sys, ":", cos_sim)
                cos_sims[i, sys] = cos_sim

                if C_dist == "_unif_C" and config.val_dataset_typ == "ypred":
                    # plot fir bounds
                    for j in range(fir_bounds.shape[1] - 2):
                        handles.extend(axs[i][sys].plot(np.array(range(config.n_positions)),
                                                        fir_bounds[sys, j] * np.ones(config.n_positions),
                                                        label="IR Analytical Length " + str(j + 1), linewidth=3,
                                                        linestyle='--', color=colors[j + 5]))

                    # plot RNN errors
                    avg, std = rnn_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_errors.shape[1])) * rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(
                        axs[i][sys].scatter(np.arange(0, 32 * 5, 5), avg_numpy, label="RNN", linewidth=1, marker='x',
                                            s=50, color=colors[len(err_lss_copy)]))
                    axs[i][sys].fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy,
                                             avg_numpy + std_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_an_errors.shape[1])) * rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(
                        axs[i][sys].scatter(np.arange(0, 251, 5), avg_an_numpy, label="RNN Analytical", linewidth=1,
                                            marker='o', s=100, color=colors[len(err_lss_copy)], zorder=10))
                    axs[i][sys].fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy,
                                             avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0],
                                             alpha=0.2)

                del err_lss_copy  # Delete the variable
                gc.collect()  # Start the garbage collector

                axs[i][sys].legend(fontsize=18, loc="upper right", ncol=math.floor(len(handles) / 4))
                axs[i][sys].set_xlabel("t", fontsize=30)
                axs[i][sys].set_ylabel("Prediction Error", fontsize=30)
                axs[i][sys].grid(which="both")
                axs[i][sys].tick_params(axis='both', which='major', labelsize=30)
                axs[i][sys].tick_params(axis='both', which='minor', labelsize=20)
                axs[i][sys].set_title("KF system " + str(i) + " testing on system " + str(sys) + (
                    ": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                        ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else ": Dense A ")) + (
                                          "Uniform C" if C_dist == "_unif_C" else (
                                              "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                axs[i][sys].set_ylim(bottom=10 ** (-0.7), top=2 * 10 ** (0))
                # axs[i][sys].set_xlim(left=0, right=10)

            if sys == num_systems - 1 and i == num_systems - 1:
                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                deg_fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + (
                    "-changing_deg_kf_test" if config.changing else "deg_kf_test"))
        else:
            fig = plt.figure(figsize=(15, 9))
            ax = fig.add_subplot(111)
            # plot transformer, KF and FIR errors
            handles, err_rat = plot_errs(colors, sys, err_lss_load, irreducible_error_load, ax=ax, shade=shade,
                                         normalized=excess)

            if C_dist == "_unif_C" and config.val_dataset_typ == "ypred":
                if excess:
                    # plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)),
                                               (fir_bounds[sys, i] - irreducible_error_load[sys]) * np.ones(
                                                   config.n_positions),
                                               label="IR Analytical Length " + str(i + 1) + " sys: " + str(sys),
                                               linewidth=3, linestyle='--'))  # , color = colors[i + 5]))

                    # #plot RNN errors
                    # rnn_er = rnn_errors[sys].detach().numpy()
                    # kalman_err = err_lss_load["Kalman"][sys,:, ::5].mean(axis=(0))
                    # #figure out how to take median and quantiles of the rnn errors
                    # rnn_q1, rnn_median, rnn_q3 = np.quantile((rnn_er -kalman_err), [0.25, 0.5, 0.75], axis=-2)
                    # scale = rnn_median[1]
                    # rnn_median = rnn_median/scale
                    # rnn_q1 = rnn_q1/scale
                    # rnn_q3 = rnn_q3/scale
                    # N = rnn_median.shape[0]
                    # # Adjust the range of np.arange function
                    # x = np.arange(1, (N-1)*5 + 1, 5)
                    # handles.append(ax.scatter(x, rnn_median[1:], label="RNN sys: " + str(sys), linewidth=3, marker='x', s=50))#, color=colors[len(err_lss_load)]))
                    # if shade:
                    #     ax.fill_between(x, rnn_q1[1:], rnn_q3[1:], facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    print("rnn_an_errors.shape:", rnn_an_errors.shape)
                    # plot Analytical RNN errors
                    rnn_an_er = rnn_an_errors[sys].detach().numpy()
                    print("shape of err_lss_load[Kalman]:", err_lss_load["Kalman"][sys, :, ::5].shape)
                    kalman_err = err_lss_load["Kalman"][sys, :, ::5].mean(axis=(0))
                    # figure out how to take median and quantiles of the rnn errors
                    rnn_an_q1, rnn_an_median, rnn_an_q3 = np.quantile((rnn_an_er - kalman_err), [0.25, 0.5, 0.75],
                                                                      axis=-2)
                    scale = rnn_an_median[1]
                    rnn_an_median = rnn_an_median / scale
                    rnn_an_q1 = rnn_an_q1 / scale
                    rnn_an_q3 = rnn_an_q3 / scale
                    N = rnn_an_median.shape[0]
                    # Adjust the range of np.arange function
                    x = np.arange(1, (N - 1) * 5 + 1, 5)
                    handles.append(
                        ax.scatter(x, rnn_an_median[1:], label="RNN Analytical sys: " + str(sys), linewidth=1,
                                   marker='o', s=100))  # , color=colors[len(err_lss_load)]))
                    if shade:
                        ax.fill_between(x, rnn_an_q1[1:], rnn_an_q3[1:], facecolor=handles[-1].get_facecolor()[0],
                                        alpha=0.2)
                    # avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    # avg_an_numpy = avg_an.detach().numpy()
                    # std_an_numpy = std_an.detach().numpy()
                    # handles.append(ax.scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100))#, color=colors[len(err_lss_load)], zorder=10))
                    # ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)
                else:
                    # plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)),
                                               fir_bounds[sys, i] * np.ones(config.n_positions),
                                               label="IR Analytical Length " + str(i + 1), linewidth=3, linestyle='--',
                                               color=colors[i + 5]))

                    # plot RNN errors
                    print("rnn_errors.shape:", rnn_errors.shape)
                    avg, std = rnn_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_errors.shape[1])) * rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(
                        ax.scatter(np.arange(0, config.n_positions + 1, 5), avg_numpy, label="RNN", linewidth=1,
                                   marker='x', s=50, color=colors[len(err_lss_load)]))
                    ax.fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy, avg_numpy + std_numpy,
                                    facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_an_errors.shape[1])) * rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(
                        ax.scatter(np.arange(0, 251, 5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o',
                                   s=100, color=colors[len(err_lss_load)], zorder=10))
                    ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy,
                                    avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

            if excess:
                ncol = 1 if len(handles) < 4 else math.floor(len(handles) / 4)
                ax.legend(fontsize=18, loc="lower left", ncol=ncol)
                ax.set_xlabel("log(t)", fontsize=30)
                ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                # make the x axis log scale
                ax.set_xscale('log')
                # ax.set_ylim(bottom=-1, top=2*10**(-1))
                ax.set_title("System " + str(sys) + (": Rotated Diagonal (|N(0,1)| <= 0.95 Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal (Unif(-1,1) Eigs) A " if config.val_dataset_typ == "rotDiagA" else (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))))
                # ax.set_xlim(left=0, right=10)

                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(
                    parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + (
                        "-changing" if config.changing else "_excess"))
            else:
                ax.legend(fontsize=18, loc="upper right", ncol=max(1, math.floor(len(handles) / 4)))
                ax.set_xlabel("t", fontsize=30)
                ax.set_ylabel("Prediction Error", fontsize=30)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                ax.set_ylim(bottom=10 ** (-0.7), top=3 * 10 ** (0))
                ax.set_title("System " + str(sys) + (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")), fontsize=20)
                # ax.set_xlim(left=0, right=10)

                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(
                    parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + (
                        "-changing" if config.changing else ""))

    if run_deg_kf_test:
        # Create a DataFrame from the numpy array
        # create a list of strings that correspond to the system numbers
        test_col = ["Test sys " + str(i) for i in range(num_systems)]
        train_col = ["Train sys " + str(i) for i in range(num_systems)]

        df = pd.DataFrame(cos_sims, columns=test_col, index=train_col)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Cosine Similarities of KF Predictions')

        # Create a table and save it as an image
        tbl = table(ax, df, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_cosine_similarities_deg_kf_test")

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of KF Predictions')
        # Create a table and save it as an image
        df2 = pd.DataFrame(err_ratios, columns=test_col, index=train_col)
        tbl = table(ax, df2, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_error_ratios_deg_kf_test")

        print("zero_ratios:", zero_ratios)
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of Zero Predictions')
        # Create a table and save it as an image
        df3 = pd.DataFrame(zero_ratios[0, :].reshape(1, -1), columns=test_col)
        tbl = table(ax, df3, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_zero_ratios_deg_kf_test")

    if excess:
        ncol = 1 if len(handles) < 4 else math.floor(len(handles) / 2)
        ax.legend(fontsize=14, loc="lower left", ncol=ncol)
        ax.set_xlabel("log(t)", fontsize=30)
        ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
        ax.grid(which="both")
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        # make the x axis log scale
        ax.set_xscale('log')
        # ax.set_ylim(bottom=-1, top=2*10**(-1))
        ax.set_title("System " + str(sys) + (": Rotated Diagonal (|N(0,1)| <= 0.95 Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal (Unif(-1,1) Eigs) A " if config.val_dataset_typ == "rotDiagA" else (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))))
        # ax.set_xlim(left=0, right=10)
        os.makedirs(parent_parent_dir + f"/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff" + (
            "-changing" if config.changing else "_excess"))
