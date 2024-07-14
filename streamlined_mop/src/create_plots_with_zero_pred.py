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

plt.rcParams['axes.titlesize'] = 20

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
    with open(f"../data/test_{config.dataset_typ}.pt", "rb") as f:
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
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)   # get the inputs (observations without the last one)
        # if config.dataset_typ == "drone":  # if the dataset type is drone
        #     I = np.concatenate([I, us], axis=-1)  # concatenate the inputs

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
                    padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])   # [(L + R - 1) x O_D]
                    ls = list(np.zeros((2, config.ny)))
                    ls_analytical = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)

                    rls_wentinn = CnnKF(config.ny, ir_length, ridge=1.0)
                    for i in range(config.n_positions - 1):
                        rls_wentinn.update(
                            torch.from_numpy(padded_ys[i:i + ir_length]),
                            torch.from_numpy(padded_ys[i + ir_length])
                        )

                        ls.append(rls_wentinn(torch.Tensor(padded_ys[i + 1:i + ir_length + 1])[None]).squeeze(0, 1).detach().numpy())
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
    fig.savefig(f"../figures/{config.dataset_typ}" + ("-changing" if config.changing else ""))
    plt.show()

####################################################################################################
def compute_OLS_and_OLS_analytical(config, ys, sim_objs, ir_length, err_lss): #PENDING DELETION
    preds_rls = []
    preds_rls_analytical = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _preds_rls = []
        _preds_rls_analytical = []
        for __ys in _ys:
            ls = [np.zeros(config.ny)]
            ls_analytical = [np.linalg.norm(__ys[0], axis=-1) ** 2]
            rls = RLS(config.nx, config.ny)

            #print("shape __ys:", __ys.shape)
            #print("range of for loop:", range(__ys.shape[-2] - 1))
            for i in range(_ys.shape[-2] - 1):
                if i < 3:
                    ls.append(__ys[i])
                    ls_analytical.append(np.linalg.norm(__ys[i + 1], axis=-1) ** 2)
                else:
                    if __ys[i-3:i].shape[0] == 0:
                        print("i:", i)
                    rls.add_data(__ys[i - ir_length:i].flatten(), __ys[i])
                    _cnn_rls = CnnKF(config.ny, ir_length)
                    _cnn_rls.observation_IR.data = torch.from_numpy(np.stack([_rls.mu for _rls in rls.rlss], axis=-1)
                                                                    .reshape(ir_length, config.ny, config.ny)
                                                                    .transpose(1, 0, 2)[:, ::-1].copy())
                    ls.append(rls.predict(__ys[i - (ir_length-1):i + 1].flatten()))
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
                    _cnn_rls.observation_IR.data = torch.stack([_rls.mu for _rls in rls.rlss], axis=-1).reshape(ir_length, config.ny, config.ny).transpose(1, 0, 2)[:, ::-1].copy().to(device)
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
    print("\n\n max_ir_length + 1:", max_ir_length+1)
    for ir_length in range(2, max_ir_length + 1):
        start = time.time()
        print(f"\n\nIR length: {ir_length}")
        preds_rls_wentinn = []
        preds_rls_wentinn_analytical = []
        sys_count = 0
        for sim_obj, _ys in zip(sim_objs, ys):
            _preds_rls_wentinn = []
            _preds_rls_wentinn_analytical = []
            for __ys in _ys:
                padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])   # [(L + R - 1) x O_D]
                ls = list(np.zeros((2, config.ny)))
                ls_analytical = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)

                rls_wentinn = CnnKF(config.ny, ir_length, ridge=1.0)
                for i in range(config.n_positions - 1):
                    obs_tensor = rls_wentinn.update(
                        torch.from_numpy(padded_ys[i:i + ir_length]),
                        torch.from_numpy(padded_ys[i + ir_length])
                    )

                    ls.append(rls_wentinn(torch.Tensor(padded_ys[i + 1:i + ir_length + 1])[None]).squeeze(0, 1).detach().numpy())
                    ls_analytical.append(rls_wentinn.analytical_error(sim_obj).item())

                _preds_rls_wentinn.append(ls)
                _preds_rls_wentinn_analytical.append(ls_analytical)

                if _preds_rls_wentinn_analytical[-1][50] < 0.6 and ir_length == 2 and sys_count == 2:
                        

                    print("len of _preds_rls_wentinn_analytical[-1]:", len(_preds_rls_wentinn_analytical[-1]))

                    # Inside your loop or function where you open the file
                    file_path = f"../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C/data/observation_IR_{ir_length}.pt"
                    directory = os.path.dirname(file_path)

                    # Create the directory if it does not exist
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Now, safely open the file for writing
                    with open(file_path, "wb") as f:
                        torch.save(obs_tensor, f)
                        print("\n\n\nsaved observation_IR tensor to file")
                        print("_preds_rls_wentinn_analytical[-1][50]:", _preds_rls_wentinn_analytical[-1][50])

            preds_rls_wentinn.append(_preds_rls_wentinn)
            preds_rls_wentinn_analytical.append(_preds_rls_wentinn_analytical)
            sys_count += 1

        err_lss[f"OLS_ir_{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
        #err_lss[f"OLS_analytical_ir_{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn_analytical), axis=-1) ** 2
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.array(preds_rls_wentinn_analytical)
    end = time.time()
    print("time elapsed:", (end - start)/60, "min")
    return err_lss
    
def compute_OLS_wentinn(config, ys, sim_objs, ir_length, err_lss):
    errs_rls_wentinn = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _errs_rls_wentinn = []
        for __ys in _ys:
            padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])   # [(L + R - 1) x O_D]
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


        with open('../data/numpy_three_sys' + C_dist + '/data.pkl', 'rb') as f: #load the data.pkl file for the test data
            data = pickle.load(f)
            ys = data["observation"]
            print("ys.shape:", ys.shape)
    else:
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

    n_noise = config.n_noise

    start = time.time() #start the timer for kalman filter predictions
    if run_deg_kf_test: #degenerate system KF Predictions
        #Kalman Filter Predictions
        preds_kf_list = []
        for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)):
            inner_list = []
            for __ys in _ys:
                result = apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
                inner_list.append(result)
            preds_kf_list.append(inner_list)

        preds_kf = np.array(preds_kf_list)  # get kalman filter predictions

        #create an array of zeros to hold the kalman filter predictions
        preds_kf = np.zeros((num_systems, num_systems, num_trials, config.n_positions + 1, config.ny)) #first axis is the system that the kalman filter is being trained on, second axis is the system that the kalman filter is being tested on

        errs_kf = np.zeros((num_systems, num_systems, num_trials, config.n_positions + 1)) #first axis is the system that the kalman filter is being trained on, second axis is the system that the kalman filter is being tested on
        #iterate over sim_objs
        kf_index = 0
        for sim_obj in sim_objs: #iterate over the training systems
            for sys in range(num_systems): #iterate over the test systems
                print("Kalman filter", kf_index, "testing on system", sys)
                for trial in range(num_trials):
                    preds_kf[kf_index, sys, trial,:,:] = apply_kf(sim_obj, ys[sys,trial,:-1,:], sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise)) #get the kalman filter predictions for the test system and the training system
                errs_kf[kf_index, sys] = np.linalg.norm((ys[sys] - preds_kf[kf_index, sys]), axis=-1) ** 2 #get the errors of the kalman filter predictions for the test system and the training system
            kf_index += 1

    else: #Kalman Predictions
        preds_kf = np.array([[
                apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
                for __ys in _ys
            ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
        ])  # get kalman filter predictions
        errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2
    
    end = time.time() #end the timer for kalman filter predictions
    print("time elapsed for KF Pred:", (end - start)/60, "min") #print the time elapsed for kalman filter predictions

    err_lss = collections.OrderedDict([
        ("Kalman", errs_kf),
        ("MOP", errs_tf),
        ("Zero", errs_zero)
    ])
    print("err_lss keys:", err_lss.keys())

    #Analytical Kalman Predictions
    analytical_kf = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    err_lss["Analytical_Kalman"] = analytical_kf.reshape((num_systems,1))@np.ones((1,config.n_positions))
    print("err_lss Analytical_Kalman sys 2:", err_lss["Analytical_Kalman"][2][50])

    # # OLS Wentinn
    # start = time.time() #start the timer for OLS Wentinn predictions
    # err_lss = compute_OLS_wentinn(config, ys, sim_objs, ir_length=2, err_lss=err_lss)
    # end = time.time() #end the timer for OLS Wentinn predictions
    # print("time elapsed for OLS Wentinn Pred:", (end - start)/60, "min") #print the time elapsed for OLS Wentinn predictions

    #Original OLS
    start = time.time() #start the timer for OLS predictions
    err_lss = compute_OLS_ir(config, ys, sim_objs, max_ir_length=3, err_lss=err_lss)
    end = time.time() #end the timer for OLS predictions
    print("time elapsed for OLS Pred:", (end - start)/60, "min") #print the time elapsed for OLS predictions
    #print the value of the OLS prediction for system 2 at position 50
    print("err_lss OLS_analytical_ir_2 shape:", err_lss["OLS_analytical_ir_2"].shape)
    print("err_lss OLS_analytical_ir_2 sys 2:", err_lss["OLS_analytical_ir_2"][2][:,50].mean())

    # #Revised OLS
    # print("\n\nREVISED OLS")
    # sim_obj_td = torch.stack([
    #     TensorDict({
    #         'F': torch.Tensor(sim.A),                              # [N x S_D x S_D]
    #         'H': torch.Tensor(sim.C),                              # [N x O_D x S_D]
    #         'sqrt_S_W': sim.sigma_w * torch.eye(sim.C.shape[-1]),  # [N x S_D x S_D]
    #         'sqrt_S_V': sim.sigma_v * torch.eye(sim.C.shape[-2])   # [N x O_D x O_D]
    #     }, batch_size=())
    #     for sim in sim_objs
    # ])

    # for ir_length in range(1, 4):
    #     print(f"IR length: {ir_length}")
    #     start = time.time()
    #     rls_preds, rls_analytical_error = [], []

    #     torch_ys = torch.Tensor(ys)         # [N x E x L x O_D]
    #     print("torch_ys.shape:", torch_ys.shape)
    #     padded_torch_ys = torch.cat([
    #         torch_ys,
    #         # torch.zeros((num_systems, num_trials, ir_length - 1, config.ny))
    #         torch.zeros((num_systems, num_trials, config.n_positions +1, config.ny))
    #     ], dim=-2)                                  # [N x E x (L + R - 1) x O_D]

    #     rls_wentinn = CnnKF(batch_shape=(num_systems, num_trials), ny=config.ny, ir_length=ir_length, ridge=1.0)
    #     for i in range(config.n_positions - 1):
    #         rls_wentinn.update(
    #             padded_torch_ys[:, :, i:i + ir_length],
    #             padded_torch_ys[:, :, i + ir_length]
    #         )
    #         rls_preds.append(rls_wentinn(padded_torch_ys[i + 1:i + ir_length + 1]))
    #         rls_analytical_error.append(rls_wentinn.analytical_error(sim_obj_td[:, None]))

    #     rls_preds = torch.stack(rls_preds, dim=2).detach().numpy()                      # [N x E x L x O_D]
    #     rls_analytical_error = torch.stack(rls_analytical_error, dim=2).detach.numpy()  # [N x E x L]

    #     err_lss[f"OLS_ir_length{ir_length}"] = np.linalg.norm(ys - np.array(rls_preds), axis=-1) ** 2

    #     # err_lss[f"OLS_ir_length{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
    #     end = time.time()
    #     print("time elapsed:", (end - start)/60, "min")

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


        with open('../data/numpy_three_sys' + C_dist + '/data.pkl', 'rb') as f: #load the data.pkl file for the test data
            data = pickle.load(f)
            ys = data["observation"]
            print("ys.shape:", ys.shape)
    else:
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
        ("MOP", errs_tf),
        ("Zero", errs_zero)
    ])
    print("err_lss keys:", err_lss.keys())

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    return err_lss, irreducible_error

def save_preds(run_deg_kf_test, config):
    err_lss, irreducible_error = compute_errors(config, config.C_dist, run_deg_kf_test, wentinn_data=False)  #, emb_dim)

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
    if run_deg_kf_test:
        #save err_lss and irreducible_error to a file
        with open(parent_parent_dir + "/prediction_errors" + config.C_dist + f"/{config.dataset_typ}_err_lss_deg_kf_test.pkl", "wb") as f:
            pickle.dump(err_lss, f)
    else:
        #save err_lss and irreducible_error to a file
        with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_err_lss.pkl", "wb") as f:
            pickle.dump(err_lss, f)
            print("err_lss keys", err_lss.keys())

    with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_irreducible_error.pkl", "wb") as f:
        pickle.dump(irreducible_error, f)  

def save_preds_conv(run_deg_kf_test, config):

    #get the step size from the ckpt_path
    step_size = config.ckpt_path.split("/")[-1].split("_")[-1]
    print("step_size:", step_size)

    #make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)
    print("parent_dir:", parent_dir)

    #get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    print("parent_parent_dir:", parent_parent_dir)

    # a boolean for whether the below directory exists
    if not os.path.exists(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size):
        os.makedirs(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size, exist_ok=False)
        
        err_lss, irreducible_error = compute_errors_conv(config, config.C_dist, run_deg_kf_test, wentinn_data=False)  #, emb_dim)

        #save err_lss and irreducible_error to a file
        with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_err_lss.pkl", "wb") as f:
            pickle.dump(err_lss, f)

        with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_irreducible_error.pkl", "wb") as f:
            pickle.dump(irreducible_error, f) 
        return
    else:
        print("The directory for ", step_size, " already exists") 
        return

def load_preds(run_deg_kf_test, excess, num_systems, config):

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

    if run_deg_kf_test:
        with open(parent_parent_dir + "/prediction_errors" + config.C_dist + f"/{config.dataset_typ}_err_lss_deg_kf_test.pkl", "rb") as f:
            err_lss_load = pickle.load(f)
            print("len(err_lss_load):", len(err_lss_load))
    else:
        with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_err_lss.pkl", "rb") as f:
            err_lss_load = pickle.load(f)

    print("err_lss_load keys:", err_lss_load.keys())

    with open(parent_parent_dir + "/prediction_errors" + config.C_dist + "_" + step_size + f"/{config.dataset_typ}_irreducible_error.pkl", "rb") as f:
        irreducible_error_load = pickle.load(f)
    
    print(irreducible_error_load)

    if config.C_dist == "_unif_C" and config.dataset_typ == "ypred":
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
            rnn_errors = rnn_errors.permute(1,2,0)

        with open(f"../data/wentinn_12_04_24/analytical_errors.pt", "rb") as f:
            rnn_an_errors = torch.load(f, map_location=torch.device('cpu'))
            rnn_an_errors = rnn_an_errors.permute(1,2,0)
    else:
        fir_bounds = np.zeros((num_systems, 1))
        rnn_errors = np.zeros((num_systems, 1, 32))
        rnn_an_errors = np.zeros((num_systems, 1, 32))
    
    return err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors

def setup_deg_kf_axs_arrs(num_systems):
    #create a square array of zeros that is the size of the number of systems to hold the cosine similarities
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
        axs = [[ax1, ax2, ax3],[ax4, ax5, ax6],[ax7, ax8, ax9]]
        return cos_sims, err_ratios, zero_ratios, deg_fig, axs

def create_plots(config, run_preds, run_deg_kf_test, excess, num_systems, shade):

    C_dist = config.C_dist
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)

    if run_preds:
        print("config path:", config.ckpt_path)
        save_preds(run_deg_kf_test, config) #save the predictions to a file

    #load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess, num_systems, config)
    print("err_lss_load keys:", err_lss_load.keys())

    if run_deg_kf_test:
        cos_sims, err_ratios, zero_ratios, deg_fig, axs = setup_deg_kf_axs_arrs(num_systems)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#00ced1', '#8c564b', '#e377c2', '#A80000', '#bcbd22', '#7D00BD', '#d00960']

    print("len(err_lss_load):", len(err_lss_load))
    for sys in range(len(irreducible_error_load)):
        
        if run_deg_kf_test:
            for i in range(num_systems):
                err_lss_copy = copy.deepcopy(err_lss_load)
                err_lss_copy["Kalman"] = err_lss_copy["Kalman"][i]
                
                print("KF trained on system", i, "testing on system", sys)

                #plot transformer, KF and FIR errors
                handles, err_rat = plot_errs(colors, sys, err_lss_copy, irreducible_error_load, ax=axs[i][sys], shade=True, normalized=excess)

                err_ratios[i, sys] = err_rat[0]
                zero_ratios[i, sys] = err_rat[1]

                #compute the cosine similarity between the err_lss_load["Kalman"][i][sys] and err_lss_load[i][i]
                cos_sim = np.dot(err_lss_load["Kalman"][i][sys].flatten(), err_lss_load["Kalman"][i][i].flatten()) / (np.linalg.norm(err_lss_load["Kalman"][i][sys]) * np.linalg.norm(err_lss_load["Kalman"][sys][sys]))
                print("cosine similarity between KF trained on system", i, "testing on system", sys, "and KF trained and tested on system", sys, ":", cos_sim)
                cos_sims[i, sys] = cos_sim

                if C_dist == "_unif_C" and config.dataset_typ == "ypred":
                #plot fir bounds
                    for j in range(fir_bounds.shape[1] - 2):
                        handles.extend(axs[i][sys].plot(np.array(range(config.n_positions)), fir_bounds[sys,j]*np.ones(config.n_positions), label="IR Analytical Length " + str(j + 1), linewidth=3, linestyle='--', color = colors[j + 5]))

                    #plot RNN errors
                    avg, std = rnn_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_errors.shape[1]))*rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(axs[i][sys].scatter(np.arange(0,32*5,5), avg_numpy, label="RNN", linewidth=1, marker='x', s=50, color=colors[len(err_lss_copy)]))
                    axs[i][sys].fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy, avg_numpy + std_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(axs[i][sys].scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100, color=colors[len(err_lss_copy)], zorder=10))
                    axs[i][sys].fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                del err_lss_copy  # Delete the variable
                gc.collect()  # Start the garbage collector

                axs[i][sys].legend(fontsize=18, loc="upper right", ncol= math.floor(len(handles)/4))
                axs[i][sys].set_xlabel("t", fontsize=30)
                axs[i][sys].set_ylabel("Prediction Error", fontsize=30)
                axs[i][sys].grid(which="both")
                axs[i][sys].tick_params(axis='both', which='major', labelsize=30)
                axs[i][sys].tick_params(axis='both', which='minor', labelsize=20)
                axs[i][sys].set_title("KF system " + str(i) + " testing on system " + str(sys) + (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else ": Dense A ")) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                axs[i][sys].set_ylim(bottom=10**(-0.7), top=2*10**(0))
                # axs[i][sys].set_xlim(left=0, right=10)

            if sys == num_systems - 1 and i == num_systems - 1:
                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)
                print("parent_dir:", parent_dir)

                #get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                deg_fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + ("-changing_deg_kf_test" if config.changing else "deg_kf_test"))
        else:
            fig = plt.figure(figsize=(15, 9))
            ax = fig.add_subplot(111)
            #plot transformer, KF and FIR errors
            handles, err_rat = plot_errs(colors, sys, err_lss_load, irreducible_error_load, ax=ax, shade=shade, normalized=excess)

            if C_dist == "_unif_C" and config.dataset_typ == "ypred":
                if excess:
                    #plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)), (fir_bounds[sys,i] - irreducible_error_load[sys])*np.ones(config.n_positions), label="IR Analytical Length " + str(i + 1) + " sys: " + str(sys), linewidth=3, linestyle='--'))#, color = colors[i + 5]))

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
                    #plot Analytical RNN errors
                    rnn_an_er = rnn_an_errors[sys].detach().numpy()
                    print("shape of err_lss_load[Kalman]:", err_lss_load["Kalman"][sys,:, ::5].shape)
                    kalman_err = err_lss_load["Kalman"][sys,:, ::5].mean(axis=(0))
                    #figure out how to take median and quantiles of the rnn errors
                    rnn_an_q1, rnn_an_median, rnn_an_q3 = np.quantile((rnn_an_er - kalman_err), [0.25, 0.5, 0.75], axis=-2)
                    scale = rnn_an_median[1]
                    rnn_an_median = rnn_an_median/scale
                    rnn_an_q1 = rnn_an_q1/scale
                    rnn_an_q3 = rnn_an_q3/scale
                    N = rnn_an_median.shape[0]
                    # Adjust the range of np.arange function
                    x = np.arange(1, (N-1)*5 + 1, 5)
                    handles.append(ax.scatter(x, rnn_an_median[1:], label="RNN Analytical sys: " + str(sys), linewidth=1, marker='o', s=100))#, color=colors[len(err_lss_load)]))
                    if shade:
                        ax.fill_between(x, rnn_an_q1[1:], rnn_an_q3[1:], facecolor=handles[-1].get_facecolor()[0], alpha=0.2)
                    # avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    # avg_an_numpy = avg_an.detach().numpy()
                    # std_an_numpy = std_an.detach().numpy()
                    # handles.append(ax.scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100))#, color=colors[len(err_lss_load)], zorder=10))
                    # ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)
                else:
                    #plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)), fir_bounds[sys,i]*np.ones(config.n_positions), label="IR Analytical Length " + str(i + 1), linewidth=3, linestyle='--', color = colors[i + 5]))

                    #plot RNN errors
                    print("rnn_errors.shape:", rnn_errors.shape)
                    avg, std = rnn_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_errors.shape[1]))*rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(ax.scatter(np.arange(0,config.n_positions + 1,5), avg_numpy, label="RNN", linewidth=1, marker='x', s=50, color=colors[len(err_lss_load)]))
                    ax.fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy, avg_numpy + std_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(ax.scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100, color=colors[len(err_lss_load)], zorder=10))
                    ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

            if excess:
                ncol = 1 if len(handles) < 4 else math.floor(len(handles)/4)
                ax.legend(fontsize=18, loc="lower left", ncol=ncol)
                ax.set_xlabel("log(t)", fontsize=30)
                ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                #make the x axis log scale
                ax.set_xscale('log')
                # ax.set_ylim(bottom=-1, top=2*10**(-1))
                ax.set_title("System " + str(sys)+ (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else (": N(0,0.33) A " if config.dataset_typ == "gaussA" else ": Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                # ax.set_xlim(left=0, right=10)

                #get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                #get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + ("-changing" if config.changing else "_excess"))
            else:
                ax.legend(fontsize=18, loc="upper right", ncol= math.floor(len(handles)/4))
                ax.set_xlabel("t", fontsize=30)
                ax.set_ylabel("Prediction Error", fontsize=30)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)

                ax.set_ylim(bottom=10**(-0.7), top=1*10**(2)) #set the y axis limits

                ax.set_title("System " + str(sys)+ (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else (": N(0,0.33) A " if config.dataset_typ == "gaussA" else ": Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")), fontsize=20)
                # ax.set_xlim(left=0, right=10)

                #get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                #get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + ("-changing" if config.changing else ""))

    if run_deg_kf_test:
        # Create a DataFrame from the numpy array
        # create a list of strings that correspond to the system numbers
        test_col = ["Test sys " + str(i) for i in range(num_systems)]
        train_col = ["Train sys " + str(i) for i in range(num_systems)]

        df = pd.DataFrame(cos_sims, columns=test_col, index=train_col)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Cosine Similarities of KF Predictions')

        # Create a table and save it as an image
        tbl = table(ax, df, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        #get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        #get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_cosine_similarities_deg_kf_test")

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of KF Predictions')
        # Create a table and save it as an image
        df2 = pd.DataFrame(err_ratios, columns=test_col, index=train_col)
        tbl = table(ax, df2, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_error_ratios_deg_kf_test")

        print("zero_ratios:", zero_ratios)
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of Zero Predictions')
        # Create a table and save it as an image
        df3 = pd.DataFrame(zero_ratios[0,:].reshape(1, -1), columns=test_col)
        tbl = table(ax, df3, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_zero_ratios_deg_kf_test")

    if excess:
        ncol = 1 if len(handles) < 4 else math.floor(len(handles)/2)
        ax.legend(fontsize=14, loc="lower left", ncol=ncol)
        ax.set_xlabel("log(t)", fontsize=30)
        ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
        ax.grid(which="both")
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        #make the x axis log scale
        ax.set_xscale('log')
        # ax.set_ylim(bottom=-1, top=2*10**(-1))
        ax.set_title(("Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else ("Upper Triangular A " if config.dataset_typ == "upperTriA" else ("N(0,0.33) A " if config.dataset_typ == "gaussA" else "Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
        # ax.set_xlim(left=0, right=10)
        os.makedirs(parent_parent_dir + f"/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff" + ("-changing" if config.changing else "_excess"))
    
    return None

def convergence_plots(j, config, run_preds, run_deg_kf_test, kfnorm, num_systems, shade, fig, ax, ts):
    excess = False
    C_dist = config.C_dist
    print("\n\n", "config path:", config.ckpt_path)
    if run_preds:
        print("\n\nRunning predictions")
        save_preds_conv(run_deg_kf_test, config) #save the predictions to a file
    print("\n\nLoading predictions")
    #load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess, num_systems, config)

    colors = [
        '#D32F2F', # Red
        '#C2185B', # Pink
        '#7B1FA2', # Purple
        '#512DA8', # Deep Purple
        '#303F9F', # Indigo
        '#1976D2', # Blue
        '#0288D1', # Light Blue
        '#0097A7', # Cyan
        '#00796B', # Teal
        '#388E3C', # Green
        '#689F38', # Light Green
        '#AFB42B', # Lime
        '#FBC02D', # Yellow
        '#FFA000', # Amber
        '#F57C00', # Orange
        '#E64A19', # Deep Orange
        '#5D4037', # Brown
        '#616161', # Grey
        '#455A64', # Blue Grey
        '#8E24AA', # Purple 600
        '#D81B60', # Pink 600
        '#3949AB', # Indigo 600
        '#F4511E', # Deep Orange 600
        '#6D4C41', # Brown 600
        '#1B5E20', # Dark Green
        '#33691E', # Lime Green Dark
        '#827717', # Olive
        '#F9A825', # Mustard
        '#FF6F00', # Orange Deep
        '#E65100', # Orange Dark
        '#BF360C', # Deep Orange Dark
        '#3E2723', # Deep Brown
        '#212121', # Almost Black
        '#263238', # Blue Grey Dark
        '#004D40', # Teal Dark
        '#006064', # Cyan Dark
        '#01579B', # Light Blue Dark
        '#0D47A1', # Blue Dark
        '#1A237E', # Indigo Dark
        '#311B92', # Deep Purple Dark
        '#4A148C', # Purple Dark
        '#880E4F', # Pink Dark
        '#B71C1C', # Red Dark
        '#D50000', # Red Accent
        '#C51162', # Pink Accent
        '#AA00FF', # Purple Accent
        '#6200EA', # Deep Purple Accent
        '#304FFE', # Indigo Accent
    ]
    print("\n\nPlotting predictions")
    sys_errs = []
    for sys in range(len(irreducible_error_load)):            
        #plot transformer, KF and FIR errors
        #get the checkpoint steps number from the checkpoint path
        ckpt_steps = config.ckpt_path.split("step=")[1].split(".")[0]
        print("\n\nckpt_steps:", ckpt_steps)
        handles, err_avg_t = plot_errs_conv(ts, j, colors, sys, err_lss_load, irreducible_error_load, ckpt_steps, kfnorm, ax=ax[sys], shade=shade)
        sys_errs.append(err_avg_t) #append the system number and the error average at step t
        
        
        # Step 1: Collect legend handles and labels
        handles, labels = ax[sys].get_legend_handles_labels()

        # Step 2: Sort handles and labels based on "MOP" part
        # Extracting the number after "MOP" and using it for sorting
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda hl: int(hl[1].split("MOP")[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        print("sorted labels", sorted_labels)

        # Step 3: Create the legend with sorted handles and labels
        ax[sys].legend(sorted_handles, sorted_labels, fontsize=18, loc="upper right", ncol=1)

        #ax[sys].legend(fontsize=18, loc="upper right", ncol=1)
        ax[sys].set_xlabel("t", fontsize=30)
        ax[sys].set_ylabel("Prediction Error", fontsize=30)
        ax[sys].grid(which="both")
        ax[sys].tick_params(axis='both', which='major', labelsize=30)
        ax[sys].tick_params(axis='both', which='minor', labelsize=20)

        #set y axis limits
        ax[sys].set_ylim(bottom=10**(-2), top=5*10**(1))

        ax[sys].set_title("System " + str(sys)+ (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else (": N(0,0.33) A " if config.dataset_typ == "gaussA" else ": Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")) + (" Normalized" if kfnorm else ""), fontsize=20)
        # ax.set_xlim(left=0, right=10)

    #get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    #get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
    fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_conv" + ("_normalized" if kfnorm else "") + ("-changing" if config.changing else ""))
    
    return (ckpt_steps, sys_errs) #return the checkpoint steps number and the system errors

####################################################################################################
# main function
if __name__ == '__main__':
    config = Config()

    C_dist = "_gauss_C" #"_unif_C" #"_gauss_C" #"_gauss_C_large_var"
    run_preds = True #run the predictions evaluation
    run_deg_kf_test = False #run degenerate KF test
    excess = False #run the excess plots
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)
    shade = False

    num_systems = config.num_val_tasks  # number of validation tasks
    
    if run_preds:
        print("config path:", config.ckpt_path)
        save_preds(run_deg_kf_test, config) #save the predictions to a file


    #load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess, num_systems, config)

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

                #plot transformer, KF and FIR errors
                handles, err_rat = plot_errs(colors, sys, err_lss_copy, irreducible_error_load, ax=axs[i][sys], shade=True, normalized=excess)
                print("err_rat:", err_rat)

                err_ratios[i, sys] = err_rat[0]
                zero_ratios[i, sys] = err_rat[1]

                #compute the cosine similarity between the err_lss_load["Kalman"][i][sys] and err_lss_load[i][i]
                cos_sim = np.dot(err_lss_load["Kalman"][i][sys].flatten(), err_lss_load["Kalman"][i][i].flatten()) / (np.linalg.norm(err_lss_load["Kalman"][i][sys]) * np.linalg.norm(err_lss_load["Kalman"][sys][sys]))
                print("cosine similarity between KF trained on system", i, "testing on system", sys, "and KF trained and tested on system", sys, ":", cos_sim)
                cos_sims[i, sys] = cos_sim

                if C_dist == "_unif_C" and config.dataset_typ == "ypred":
                #plot fir bounds
                    for j in range(fir_bounds.shape[1] - 2):
                        handles.extend(axs[i][sys].plot(np.array(range(config.n_positions)), fir_bounds[sys,j]*np.ones(config.n_positions), label="IR Analytical Length " + str(j + 1), linewidth=3, linestyle='--', color = colors[j + 5]))

                    #plot RNN errors
                    avg, std = rnn_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_errors.shape[1]))*rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(axs[i][sys].scatter(np.arange(0,32*5,5), avg_numpy, label="RNN", linewidth=1, marker='x', s=50, color=colors[len(err_lss_copy)]))
                    axs[i][sys].fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy, avg_numpy + std_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(axs[i][sys].scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100, color=colors[len(err_lss_copy)], zorder=10))
                    axs[i][sys].fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                del err_lss_copy  # Delete the variable
                gc.collect()  # Start the garbage collector

                axs[i][sys].legend(fontsize=18, loc="upper right", ncol= math.floor(len(handles)/4))
                axs[i][sys].set_xlabel("t", fontsize=30)
                axs[i][sys].set_ylabel("Prediction Error", fontsize=30)
                axs[i][sys].grid(which="both")
                axs[i][sys].tick_params(axis='both', which='major', labelsize=30)
                axs[i][sys].tick_params(axis='both', which='minor', labelsize=20)
                axs[i][sys].set_title("KF system " + str(i) + " testing on system " + str(sys) + (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else ": Dense A ")) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                axs[i][sys].set_ylim(bottom=10**(-0.7), top=2*10**(0))
                # axs[i][sys].set_xlim(left=0, right=10)

            if sys == num_systems - 1 and i == num_systems - 1:
                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)
                print("parent_dir:", parent_dir)

                #get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                deg_fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + ("-changing_deg_kf_test" if config.changing else "deg_kf_test"))
        else:
            fig = plt.figure(figsize=(15, 9))
            ax = fig.add_subplot(111)
            #plot transformer, KF and FIR errors
            handles, err_rat = plot_errs(colors, sys, err_lss_load, irreducible_error_load, ax=ax, shade=shade, normalized=excess)

            if C_dist == "_unif_C" and config.dataset_typ == "ypred":
                if excess:
                    #plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)), (fir_bounds[sys,i] - irreducible_error_load[sys])*np.ones(config.n_positions), label="IR Analytical Length " + str(i + 1) + " sys: " + str(sys), linewidth=3, linestyle='--'))#, color = colors[i + 5]))

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
                    #plot Analytical RNN errors
                    rnn_an_er = rnn_an_errors[sys].detach().numpy()
                    print("shape of err_lss_load[Kalman]:", err_lss_load["Kalman"][sys,:, ::5].shape)
                    kalman_err = err_lss_load["Kalman"][sys,:, ::5].mean(axis=(0))
                    #figure out how to take median and quantiles of the rnn errors
                    rnn_an_q1, rnn_an_median, rnn_an_q3 = np.quantile((rnn_an_er - kalman_err), [0.25, 0.5, 0.75], axis=-2)
                    scale = rnn_an_median[1]
                    rnn_an_median = rnn_an_median/scale
                    rnn_an_q1 = rnn_an_q1/scale
                    rnn_an_q3 = rnn_an_q3/scale
                    N = rnn_an_median.shape[0]
                    # Adjust the range of np.arange function
                    x = np.arange(1, (N-1)*5 + 1, 5)
                    handles.append(ax.scatter(x, rnn_an_median[1:], label="RNN Analytical sys: " + str(sys), linewidth=1, marker='o', s=100))#, color=colors[len(err_lss_load)]))
                    if shade:
                        ax.fill_between(x, rnn_an_q1[1:], rnn_an_q3[1:], facecolor=handles[-1].get_facecolor()[0], alpha=0.2)
                    # avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    # avg_an_numpy = avg_an.detach().numpy()
                    # std_an_numpy = std_an.detach().numpy()
                    # handles.append(ax.scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100))#, color=colors[len(err_lss_load)], zorder=10))
                    # ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)
                else:
                    #plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)), fir_bounds[sys,i]*np.ones(config.n_positions), label="IR Analytical Length " + str(i + 1), linewidth=3, linestyle='--', color = colors[i + 5]))

                    #plot RNN errors
                    print("rnn_errors.shape:", rnn_errors.shape)
                    avg, std = rnn_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_errors.shape[1]))*rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(ax.scatter(np.arange(0,config.n_positions + 1,5), avg_numpy, label="RNN", linewidth=1, marker='x', s=50, color=colors[len(err_lss_load)]))
                    ax.fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy, avg_numpy + std_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(ax.scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100, color=colors[len(err_lss_load)], zorder=10))
                    ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

            if excess:
                ncol = 1 if len(handles) < 4 else math.floor(len(handles)/4)
                ax.legend(fontsize=18, loc="lower left", ncol=ncol)
                ax.set_xlabel("log(t)", fontsize=30)
                ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                #make the x axis log scale
                ax.set_xscale('log')
                # ax.set_ylim(bottom=-1, top=2*10**(-1))
                ax.set_title("System " + str(sys)+ (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else (": N(0,0.33) A " if config.dataset_typ == "gaussA" else ": Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                # ax.set_xlim(left=0, right=10)

                #get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                #get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + ("-changing" if config.changing else "_excess"))
            else:
                ax.legend(fontsize=18, loc="upper right", ncol= math.floor(len(handles)/4))
                ax.set_xlabel("t", fontsize=30)
                ax.set_ylabel("Prediction Error", fontsize=30)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                ax.set_ylim(bottom=10**(-0.7), top=3*10**(0))
                ax.set_title("System " + str(sys)+ (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else (": N(0,0.33) A " if config.dataset_typ == "gaussA" else ": Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")), fontsize=20)
                # ax.set_xlim(left=0, right=10)

                #get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                #get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + ("-changing" if config.changing else ""))

    if run_deg_kf_test:
        # Create a DataFrame from the numpy array
        # create a list of strings that correspond to the system numbers
        test_col = ["Test sys " + str(i) for i in range(num_systems)]
        train_col = ["Train sys " + str(i) for i in range(num_systems)]

        df = pd.DataFrame(cos_sims, columns=test_col, index=train_col)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Cosine Similarities of KF Predictions')

        # Create a table and save it as an image
        tbl = table(ax, df, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        #get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        #get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_cosine_similarities_deg_kf_test")

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of KF Predictions')
        # Create a table and save it as an image
        df2 = pd.DataFrame(err_ratios, columns=test_col, index=train_col)
        tbl = table(ax, df2, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_error_ratios_deg_kf_test")

        print("zero_ratios:", zero_ratios)
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of Zero Predictions')
        # Create a table and save it as an image
        df3 = pd.DataFrame(zero_ratios[0,:].reshape(1, -1), columns=test_col)
        tbl = table(ax, df3, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_zero_ratios_deg_kf_test")

    if excess:
        ncol = 1 if len(handles) < 4 else math.floor(len(handles)/2)
        ax.legend(fontsize=14, loc="lower left", ncol=ncol)
        ax.set_xlabel("log(t)", fontsize=30)
        ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
        ax.grid(which="both")
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        #make the x axis log scale
        ax.set_xscale('log')
        # ax.set_ylim(bottom=-1, top=2*10**(-1))
        ax.set_title(("Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else ("Upper Triangular A " if config.dataset_typ == "upperTriA" else ("N(0,0.33) A " if config.dataset_typ == "gaussA" else "Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
        # ax.set_xlim(left=0, right=10)
        os.makedirs(parent_parent_dir + f"/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + C_dist + "_system_cutoff" + ("-changing" if config.changing else "_excess"))
