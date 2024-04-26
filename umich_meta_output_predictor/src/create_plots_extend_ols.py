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
import time

from core import Config
from dyn_models import apply_kf, generate_lti_sample, generate_changing_lti_sample, generate_drone_sample, \
    apply_ekf_drone
from models import GPT2, CnnKF
from utils import RLS, plot_errs
import pickle
import math

plt.rcParams['axes.titlesize'] = 40

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
    end = time.time()


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
    fig.savefig(f"../figures/{config.dataset_typ}" + ("-changing" if config.changing else "_extend"))
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
        with open(f"../data/val_{config.dataset_typ}.pkl", "rb") as f:
            samples = pickle.load(f)
            # for every 2000 entries in samples, get the observation values and append them to the ys list
            i = 0
            ys = np.zeros((num_systems, num_trials, config.n_positions + 1, config.ny))
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
        with open(f"../data/val_{config.dataset_typ}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)
            # print("type of sim_objs:", type(sim_objs))
            # print("shape of sim_objs:", sim_objs.shape)
            # print("len sim_objs", len(sim_objs))

        # raise Exception("Just checking the shape of ys")
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2     # get the errors of zero predictions
    # print("errs_tf.shape:", errs_tf.shape)

    n_noise = config.n_noise

    # if config.dataset_typ == "drone":
    #     preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])
    kalman_time = 0
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
        start = time.time()
        preds_kf = np.array([[
                apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
                for __ys in _ys
            ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
        ])  # get kalman filter predictions
        errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2
        kalman_time = end - start
        print("Time taken for Kalman filter:", kalman_time)

    err_lss = collections.OrderedDict([
        ("Kalman", errs_kf),
        ("Zero", errs_zero)
    ])

    analytical_kf = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    print("analytical_kf.shape:", analytical_kf.shape)
    err_lss["Analytical_Kalman"] = analytical_kf.reshape((num_systems,1))@np.ones((1,config.n_positions))
    print("err_lss[Analytical_Kalman].shape:", err_lss["Analytical_Kalman"].shape)


    ir_length = 2
    ols_times = []
    # Debugging implemented OLS
    errs_rls_wentinn = []
    start = time.time()
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
    end = time.time()
    print("Time taken for OLS_wentinn:", end - start)
    ols_times.append(end - start)


    for ir_length in range(1, 4):
        start = time.time()
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
        end = time.time()
        print("Time taken for OLS_ir_length" + str(ir_length) + ":", end - start)
        ols_times.append(end - start)

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    return err_lss, irreducible_error, ols_times, kalman_time

####################################################################################################
# main function
if __name__ == '__main__':
    config = Config()

    C_dist = "_gauss_C" #"_unif_C" #"_gauss_C" #"_gauss_C_large_var"
    run_preds = False #run the predictions evaluation
    run_deg_kf_test = False #run degenerate KF test
    excess = True #run the excess plots
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)
    shade = False

    num_systems = config.num_val_tasks  # number of validation tasks
    
    if run_preds:
        err_lss, irreducible_error, ols_times, kalman_time = compute_errors(config, C_dist, run_deg_kf_test, wentinn_data=False)#, emb_dim)

        #make the prediction errors directory
        os.makedirs("../data/prediction_errors" + C_dist, exist_ok=True)
        if run_deg_kf_test:
            #save err_lss and irreducible_error to a file
            with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_deg_kf_test_extend_ols.pkl", "wb") as f:
                pickle.dump(err_lss, f)
        else:
            #save err_lss and irreducible_error to a file
            with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_extend_ols.pkl", "wb") as f:
                pickle.dump(err_lss, f)

        # with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_irreducible_error.pkl", "wb") as f:
        #     pickle.dump(irreducible_error, f)

    #load the prediction errors from the file

    if run_deg_kf_test:
        with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_deg_kf_test_extend_ols.pkl", "rb") as f:
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
        with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss_extend_ols.pkl", "rb") as f:
            err_lss_load_extend = pickle.load(f)
            print("shape of analytical kalman error:", err_lss_load_extend["Analytical_Kalman"].shape)
            if excess == True:
                err_lss_load_extend["Analytical_Kalman"] = np.append(err_lss_load_extend["Analytical_Kalman"],err_lss_load_extend["Analytical_Kalman"][:,-1, np.newaxis], axis=-1)
            print("shape of analytical kalman error:", err_lss_load_extend["Analytical_Kalman"].shape)
        with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_err_lss.pkl", "rb") as f:
            err_lss_load = pickle.load(f)
            print("shape of analytical kalman error:", err_lss_load["Analytical_Kalman"].shape)
            if excess == True:
                err_lss_load["Analytical_Kalman"] = np.append(err_lss_load["Analytical_Kalman"],err_lss_load["Analytical_Kalman"][:,-1, np.newaxis], axis=-1)
            print("shape of analytical kalman error:", err_lss_load["Analytical_Kalman"].shape)
    with open("../data/prediction_errors" + C_dist + f"/{config.dataset_typ}_irreducible_error.pkl", "rb") as f:
        irreducible_error_load = pickle.load(f)
    
    print(irreducible_error_load)

    if C_dist == "_unif_C" and config.dataset_typ == "ypred":
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
                axs[i][sys].set_title("KF system " + str(i) + " testing on system " + str(sys) + (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else (": N(0,0.33) A " if config.dataset_typ == "gaussA" else ": Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                axs[i][sys].set_ylim(bottom=10**(-0.7), top=2*10**(0))
                # axs[i][sys].set_xlim(left=0, right=10)

            if sys == num_systems - 1 and i == num_systems - 1:
                os.makedirs("../figures", exist_ok=True)
                deg_fig.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + ("-changing_deg_kf_test" if config.changing else "deg_kf_test_extend"))
        else:
            if not excess:
                fig = plt.figure(figsize=(15, 9))
                ax = fig.add_subplot(111)
            #plot transformer, KF and FIR errors
            handles, err_rat = plot_errs(colors, sys, err_lss_load, irreducible_error_load, ax=ax, shade=shade, normalized=excess)

            handles, err_rat = plot_errs(colors, sys, err_lss_load_extend, irreducible_error_load, ax=ax, shade=shade, normalized=excess)

            if C_dist == "_unif_C" and config.dataset_typ == "ypred":
                if excess:
                    #plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)), (fir_bounds[sys,i] - irreducible_error_load[sys])*np.ones(config.n_positions), label="IR Analytical Length " + str(i + 1) + " sys: " + str(sys), linewidth=3, linestyle='--'))#, color = colors[i + 5]))

                    #plot RNN errors
                    rnn_er = rnn_errors[sys].detach().numpy()
                    kalman_err = err_lss_load["Kalman"][sys,:, ::5].mean(axis=(0))
                    #figure out how to take median and quantiles of the rnn errors
                    rnn_q1, rnn_median, rnn_q3 = np.quantile((rnn_er -kalman_err), [0.25, 0.5, 0.75], axis=-2)
                    N = rnn_median.shape[0]
                    # Adjust the range of np.arange function
                    x = np.arange(1, (N-1)*5 + 1, 5)
                    handles.append(ax.scatter(x, rnn_median[1:], label="RNN sys: " + str(sys), linewidth=3, marker='x', s=50))#, color=colors[len(err_lss_load)]))
                    if shade:
                        ax.fill_between(x, rnn_q1[1:], rnn_q3[1:], facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    print("rnn_an_errors.shape:", rnn_an_errors.shape)
                    #plot RNN errors
                    rnn_an_er = rnn_an_errors[sys].detach().numpy()
                    print("shape of err_lss_load[Kalman]:", err_lss_load["Kalman"][sys,:, ::5].shape)
                    kalman_err = err_lss_load["Kalman"][sys,:, ::5].mean(axis=(0))
                    #figure out how to take median and quantiles of the rnn errors
                    rnn_an_q1, rnn_an_median, rnn_an_q3 = np.quantile((rnn_an_er - kalman_err), [0.25, 0.5, 0.75], axis=-2)
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
                continue
                # ncol = 1 if len(handles) < 4 else math.floor(len(handles)/4)
                # ax.legend(fontsize=18, loc="lower left", ncol=ncol)
                # ax.set_xlabel("log(t)", fontsize=30)
                # ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
                # ax.grid(which="both")
                # ax.tick_params(axis='both', which='major', labelsize=30)
                # ax.tick_params(axis='both', which='minor', labelsize=20)
                # #make the x axis log scale
                # ax.set_xscale('log')
                # # ax.set_ylim(bottom=-1, top=2*10**(-1))
                # ax.set_title("System " + str(sys)+ (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else (": Upper Triangular A " if config.dataset_typ == "upperTriA" else ": Dense A ")) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                # # ax.set_xlim(left=0, right=10)

                # os.makedirs("../figures", exist_ok=True)
                # fig.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + ("-changing" if config.changing else "_excess"))
            else:
                ax.legend(fontsize=18, loc="upper right", ncol= math.floor(len(handles)/4))
                ax.set_xlabel("t", fontsize=30)
                ax.set_ylabel("Prediction Error", fontsize=30)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                ax.set_ylim(bottom=10**(-0.7), top=2*10**(0))
                ax.set_title("System " + str(sys)+ (": Rotated Diagonal A " if config.dataset_typ == "rotDiagA" else ("Upper Triangular A " if config.dataset_typ == "upperTriA" else (": N(0,0.33) A " if config.dataset_typ == "gaussA" else ": Dense A "))) + ("Uniform C" if C_dist == "_unif_C" else ("N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")), fontsize=20)
                # ax.set_xlim(left=0, right=10)

                os.makedirs("../figures", exist_ok=True)
                fig.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + ("-changing" if config.changing else "_extend"))

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
        plt.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_cosine_similarities_deg_kf_test_extend")

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of KF Predictions')
        # Create a table and save it as an image
        df2 = pd.DataFrame(err_ratios, columns=test_col, index=train_col)
        tbl = table(ax, df2, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_error_ratios_deg_kf_test_extend")

        print("zero_ratios:", zero_ratios)
        fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of Zero Predictions')
        # Create a table and save it as an image
        df3 = pd.DataFrame(zero_ratios[0,:].reshape(1, -1), columns=test_col)
        tbl = table(ax, df3, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_zero_ratios_deg_kf_test_extend")

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
        fig.savefig(f"../figures/{config.dataset_typ}" + C_dist + "_system_cutoff" + ("-changing" if config.changing else "_excess_extend"))

             