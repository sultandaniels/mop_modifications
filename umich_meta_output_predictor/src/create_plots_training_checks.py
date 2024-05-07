import collections
import logging
import os
import copy
import pandas as pd
from pandas.plotting import table

import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

from core import Config
from dyn_models import apply_kf, generate_lti_sample, generate_changing_lti_sample, generate_drone_sample, \
    apply_ekf_drone
from models import GPT2, CnnKF
from utils import RLS, plot_errs
import pickle
import math
from tensordict import TensorDict
from striprtf.striprtf import rtf_to_text

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

def compute_errors(config, C_dist, run_deg_kf_test, wentinn_data):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger
    config.parse_args()  # parse the arguments

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
            
        # with open("../data/val_ypred.pkl", "rb") as f:
        #         entries = pickle.load(f)
        #         print("keys of entries:", entries[0].keys())
        #         print("len of entries:", len(entries))
        #         print("shape of all the values for each key in entries[0]", {k: v.shape for k, v in entries[0].items()})
        #         #set ys equal to the observation values of all the entries
        #         ys = np.array([entry["obs"] for entry in entries])
        #         print("ys.shape:", ys.shape)
        #         # print("shape of entries:", entries["observation"].shape)
    else:
        with open(f"../data/val_{config.dataset_typ}_{config.C_dist}.pkl", "rb") as f:
            samples = pickle.load(f)
            # for every 2000 entries in samples, get the observation values and append them to the ys list
            i = 0
            ys = np.zeros((num_systems, num_trials, config.n_positions + 1, config.ny))
            # print("\n\n")
            # print("samples[0][A] eigenvalues", np.linalg.eigvals(samples[0]['A']))
            # print("samples[0][C] eigenvalues", np.linalg.svd(samples[0]['C']))
            # print("\n\n")
            # print("samples[2050][A] eigenvalues", np.linalg.eigvals(samples[2050]['A']))
            # print("samples[2050][C] eigenvalues", np.linalg.svd(samples[2050]['C']))
            # print("\n\n")
            # # print("samples[4050][A] eigenvalues", np.linalg.eigvals(samples[4050]['A']))
            # # print("samples[0][A] eigenvalues", np.linalg.eigvals(samples[0]['A']))
            # # print("samples[2050][A] eigenvalues", np.linalg.eigvals(samples[2050]['A']))
            # print("\n\n")
            # print("samples[4050][A] eigenvalues", np.linalg.eigvals(samples[4050]['A']))
            # print("samples[4050][C] eigenvalues", np.linalg.svd(samples[4050]['C']))
            
            for entry in samples:
                ys[math.floor(i/num_trials), i % num_trials] = entry["obs"]
                i += 1
            ys = ys.astype(np.float32)
            # print("ys.shape:", ys.shape)
            # print("type of ys:", type(ys))
            # print("dtype of ys:", ys.dtype)
            del samples  # Delete the variable
            gc.collect()  # Start the garbage collector

        #open fsim file
        with open(f"../data/val_{config.dataset_typ}_{config.C_dist}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)
            # print(sim_objs[0])
            # print("type of sim_objs:", type(sim_objs))
            # print("shape of sim_objs:", sim_objs.shape)
            # print("len sim_objs", len(sim_objs))

        # raise Exception("Just checking the shape of ys")

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
            _, flattened_preds_tf = model.predict_step({"xs": torch.from_numpy(flattened_I).to(device)}) #.float().to(device)})    # predict using the model
            # print("flattened_preds_tf:", flattened_preds_tf)
            preds_tf = np.reshape(flattened_preds_tf["preds"].cpu().numpy(), (*batch_shape, *I.shape[-2:])) # get the predictions
            # print("preds_tf.shape:", preds_tf.shape)
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf], axis=-2)  # concatenate the predictions
            # print("preds_tf.shape:", preds_tf.shape)

    # print("preds_tf.shape:", preds_tf.shape)
    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2     # get the errors of transformer predictions
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2     # get the errors of zero predictions
    # print("errs_tf.shape:", errs_tf.shape)

    n_noise = config.n_noise

    # if config.dataset_typ == "drone":
    #     preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])

    if run_deg_kf_test:
        # preds_kf_list = []
        # for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)):
        #     inner_list = []
        #     for __ys in _ys:
        #         result = apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
        #         inner_list.append(result)
        #     preds_kf_list.append(inner_list)

        # preds_kf = np.array(preds_kf_list)  # get kalman filter predictions

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
        # errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2

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

    analytical_kf = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    print("analytical_kf.shape:", analytical_kf.shape)
    err_lss["Analytical_Kalman"] = analytical_kf.reshape((num_systems,1))@np.ones((1,config.n_positions))
    print("err_lss[Analytical_Kalman].shape:", err_lss["Analytical_Kalman"].shape)


    ir_length = 2
    # if config.dataset_typ != "drone":
    #     preds_rls = []
    #     preds_rls_analytical = []
    #     for sim_obj, _ys in zip(sim_objs, ys):
    #         _preds_rls = []
    #         _preds_rls_analytical = []
    #         for __ys in _ys:
    #             ls = [np.zeros(config.ny)]
    #             ls_analytical = [np.linalg.norm(__ys[0], axis=-1) ** 2]

    #             rls = RLS(config.nx, config.ny)
    #             for i in range(_ys.shape[-2] - 1):
    #                 if i < 2:
    #                     ls.append(__ys[i])
    #                     ls_analytical.append(np.linalg.norm(__ys[i + 1], axis=-1) ** 2)
    #                 else:
    #                     rls.add_data(__ys[i - 2:i].flatten(), __ys[i])
    #                     _cnn_rls = CnnKF(config.ny, ir_length)
    #                     _cnn_rls.observation_IR.data = torch.from_numpy(np.stack([_rls.mu for _rls in rls.rlss], axis=-1)
    #                                                                     .reshape(ir_length, config.ny, config.ny)
    #                                                                     .transpose(1, 0, 2)[:, ::-1].copy())

    #                     ls.append(rls.predict(__ys[i - 1:i + 1].flatten()))
    #                     ls_analytical.append(_cnn_rls.analytical_error(sim_obj).item())

    #             _preds_rls.append(ls)
    #             _preds_rls_analytical.append(ls_analytical)

    #         preds_rls.append(_preds_rls)
    #         preds_rls_analytical.append(_preds_rls_analytical)

        # err_lss["OLS"] = np.linalg.norm(ys - np.array(preds_rls), axis=-1) ** 2
        # err_lss["OLS_analytical"] = np.array(preds_rls_analytical)

        # # Debugging implemented OLS
        # errs_rls_wentinn = []
        # for sim_obj, _ys in zip(sim_objs, ys):
        #     _errs_rls_wentinn = []
        #     for __ys in _ys:
        #         padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])   # [(L + R - 1) x O_D]
        #         ls = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)
        #         rls_wentinn = CnnKF(config.ny, ir_length)
        #         for i in range(config.n_positions - 1):
        #             rls_wentinn.update(
        #                 torch.from_numpy(padded_ys[i:i + ir_length]),
        #                 torch.from_numpy(padded_ys[i + ir_length])
        #             )
        #             ls.append(rls_wentinn.analytical_error(sim_obj).item())
        #         _errs_rls_wentinn.append(ls)
        #     errs_rls_wentinn.append(_errs_rls_wentinn)
        # err_lss["OLS_wentinn"] = np.array(errs_rls_wentinn)
    # for ir_length in range(1, 4):
    #     print(f"IR length: {ir_length}")
    #     preds_rls_wentinn = []
    #     preds_rls_wentinn_analytical = []
    #     for sim_obj, _ys in zip(sim_objs, ys):
    #         _preds_rls_wentinn = []
    #         _preds_rls_wentinn_analytical = []
    #         for __ys in _ys:
    #             padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])   # [(L + R - 1) x O_D]
    #             ls = list(np.zeros((2, config.ny)))
    #             ls_analytical = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)

    #             rls_wentinn = CnnKF(config.ny, ir_length, ridge=1.0)
    #             for i in range(config.n_positions - 1):
    #                 rls_wentinn.update(
    #                     torch.from_numpy(padded_ys[i:i + ir_length]),
    #                     torch.from_numpy(padded_ys[i + ir_length])
    #                 )

    #                 ls.append(rls_wentinn(torch.Tensor(padded_ys[i + 1:i + ir_length + 1])[None]).squeeze(0, 1).detach().numpy())
    #                 ls_analytical.append(rls_wentinn.analytical_error(sim_obj).item())

    #             _preds_rls_wentinn.append(ls)
    #             _preds_rls_wentinn_analytical.append(ls_analytical)

    #         preds_rls_wentinn.append(_preds_rls_wentinn)
    #         preds_rls_wentinn_analytical.append(_preds_rls_wentinn_analytical)

    #     err_lss[f"OLS_ir_length{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
    # sim_obj_td = torch.stack([
    #     TensorDict({
    #         'F': torch.Tensor(sim.A),                                                   # [N x S_D x S_D]
    #         'H': torch.Tensor(sim.C),                                                   # [N x O_D x S_D]
    #         'sqrt_S_W': sim.sigma_w * torch.eye(sim.C.shape[-1]),                       # [N x S_D x S_D]
    #         'sqrt_S_V': sim.sigma_v * torch.eye(sim.C.shape[-2])                        # [N x O_D x O_D]
    #     }, batch_size=())
    #     for sim in sim_objs
    # ])

    # for ir_length in range(1, 4):
    #     print(f"IR length: {ir_length}")
    #     rls_preds, rls_analytical_error = [], []

    #     torch_ys = torch.Tensor(ys)         # [N x E x L x O_D]
    #     padded_torch_ys = torch.cat([
    #         torch_ys,
    #         torch.zeros((num_systems, num_trials, ir_length - 1, config.ny))
    #     ])                                  # [N x E x (L + R - 1) x O_D]

    #     rls_wentinn = CnnKF((num_systems, num_trials), config.ny, ir_length, ridge=1.0)
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

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    return err_lss, irreducible_error

####################################################################################################
# main function
if __name__ == '__main__':
    config = Config()

    C_dist = config.C_dist #"_unif_C" #"_gauss_C" #"_gauss_C_large_var"
    print("C_dist:", C_dist)
    run_preds = True #run the predictions evaluation
    run_deg_kf_test = False #run degenerate KF test
    excess = False #run the excess plots
    context_el = 200 #element of the context
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)
    shade = False

    num_systems = config.num_val_tasks  # number of validation tasks
    
    if run_preds:
        all_err_lss = []
        all_irreducible_error = []
        for steps in range(4000, 40001, 4000):

            # CHANGE THE CKPT PATH ########################################
            config.override("ckpt_path", "../outputs/GPT2/240505_102734.114a39/checkpoints/batch_size_28_con_len_250_step=" + str(steps) + ".ckpt")
            print("config.ckpt_path:", config.ckpt_path)

            # Get the directory of the checkpoint file
            ckpt_dir = os.path.dirname(config.ckpt_path)

            # Get the parent directory
            parent_dir = os.path.dirname(ckpt_dir)

            print(parent_dir)

            print("\n\nReading the README file\n\n")

            # Open the README file and print its contents
            with open(parent_dir + "/README.rtf", "r") as file:
                rtf_text = file.read()

            plain_text = rtf_to_text(rtf_text)
            print(plain_text)



            err_lss, irreducible_error = compute_errors(config, C_dist, run_deg_kf_test, wentinn_data=False)
            all_err_lss.append(err_lss)
            all_irreducible_error.append(irreducible_error)
            print("Computed prediction errors for step", steps)
            print("\n\n")

        #make the prediction errors directory
        os.makedirs("../data/prediction_errors" + C_dist, exist_ok=True)
        if run_deg_kf_test:
            #save err_lss and irreducible_error to a file
            with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_deg_kf_test_checks.pkl", "wb") as f:
                pickle.dump(all_err_lss, f)
        else:
            #save err_lss and irreducible_error to a file
            with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_checks.pkl", "wb") as f:
                pickle.dump(all_err_lss, f)

        with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_irreducible_error_checks.pkl", "wb") as f:
            pickle.dump(all_irreducible_error, f)
        print("Saved prediction errors to file\n\n\n")

    #load the prediction errors from the file

    if run_deg_kf_test:
        with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_deg_kf_test.pkl", "rb") as f:
            err_lss_load = pickle.load(f)
            print("len(err_lss_load):", len(err_lss_load))
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
    else:
        with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_checks.pkl", "rb") as f:
            err_lss_load_checks = pickle.load(f)
            # if excess == True:
            #     err_lss_load["Analytical_Kalman"] = np.append(err_lss_load["Analytical_Kalman"],err_lss_load["Analytical_Kalman"][:,-1, np.newaxis], axis=-1)
            # irreducible_error = [err_lss_load["Analytical_Kalman"][i][i] for i in range(num_systems)]
            # print("irreducible_error:", irreducible_error)
            # with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_irreducible_error_checks.pkl", "wb") as f:
            #     pickle.dump(irreducible_error, f)

    with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_irreducible_error_checks.pkl", "rb") as f:
        irreducible_error_load_checks = pickle.load(f)

    print("loaded prediction errors from file\n\n\n")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#A80000', '#bcbd22']

    # if not excess:
    #     fig = plt.figure(figsize=(15, 9))
    #     ax = fig.add_subplot(111)

    error_vals = np.zeros((num_systems, 10))
    std_devs = np.zeros((num_systems, 10))
    print("num_systems:", num_systems)
    print("error_vals.shape:", error_vals.shape)

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    print("len(err_lss_load):", len(err_lss_load_checks))
    i = 0
    for err_lss_load in err_lss_load_checks:
        irreducible_error_load = irreducible_error_load_checks[i]
        for sys in range(len(irreducible_error_load)):
            #plot transformer, KF and FIR errors
            print("shape of err_lss_load[MOP]:", err_lss_load["MOP"][sys,:,context_el].shape)
            error_vals[sys,i] = err_lss_load["MOP"][sys,:,context_el].mean()
            std_devs[sys,i] = err_lss_load["MOP"][sys,:,context_el].std()*(3/np.sqrt(err_lss_load["MOP"][sys,:,context_el].shape[0])) #3 standard deviations divided by the square root of the number of samples
        i += 1

    print("err_lss_checks[Analytical_Kalman]:", err_lss_load_checks[0]["Analytical_Kalman"].shape)
    print("irreducible_error_load_checks]:", irreducible_error_load_checks)
# for each system, plot error_vals with std dev error bars. The system number is the first axis of error_vals
    for sys in range(num_systems):
        ax.errorbar(4000*np.arange(10), error_vals[sys], yerr=std_devs[sys], label=f"System {sys}", color=colors[sys])
        ax.plot(4000*np.arange(10), irreducible_error_load_checks[0][sys]*np.ones(10), label=f"System {sys} Irreducible Error", color=colors[sys], linestyle="--")
    ax.legend()
    plt.title(("Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else ("Upper Triangular A " if config.dataset_typ == "upperTriA" else ("N(0,0.33) A " if config.dataset_typ == "gaussA" else "Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")) + " Test Error vs Training Step")
    plt.xlabel("Training Step")
    plt.ylabel("Test Error")
    #make a caption for the plot that says the context element
    plt.figtext(0.5, 0.01, f"Context element that was tested: {context_el}", wrap=True, horizontalalignment='center', fontsize=12)
    os.makedirs("../figures/training_step", exist_ok=True)
    fig.savefig(f"../figures/training_step/{config.dataset_typ}" + C_dist + "_test_error_vs_training_step_context_el_" + str(context_el))


####################################################################################################
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

        os.makedirs("../figures", exist_ok=True)
        plt.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_cosine_similarities_deg_kf_test")

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of KF Predictions')
        # Create a table and save it as an image
        df2 = pd.DataFrame(err_ratios, columns=test_col, index=train_col)
        tbl = table(ax, df2, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_error_ratios_deg_kf_test")

        print("zero_ratios:", zero_ratios)
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of Zero Predictions')
        # Create a table and save it as an image
        df3 = pd.DataFrame(zero_ratios[0,:].reshape(1, -1), columns=test_col)
        tbl = table(ax, df3, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_zero_ratios_deg_kf_test")

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
        os.makedirs("../figures", exist_ok=True)
        fig.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_system_cutoff" + ("-changing" if config.changing else "_excess"))