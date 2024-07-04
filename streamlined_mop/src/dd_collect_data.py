import logging
from dyn_models import generate_lti_sample, generate_lti_sample_new_eig, FilterSim
from core import Config
from tqdm import tqdm
import pickle
import os
import numpy as np
from models import GPT2

def collect_data(config, output_dirs):
    config.parse_args() # Not sure if this is needed
    
    Ms = sorted(config.Ms)
    systems = []
    for i in range(Ms[-1]):
        A = FilterSim.construct_A(config.dataset_typ, config.nx)
        C = FilterSim.construct_C(A, config.ny, normC=True)
        systems.append({"A": A, "C": C})
    
    for M in Ms:
        samples = []
        sim_objs = []

        for i in tqdm(range(M)):
            num_traces = config.total_training_traces // M #number of traces per system must depend on number of systems to have constant total number of traces
            fsim, sample = generate_lti_sample(config.C_dist, config.dataset_typ, num_traces, config.n_positions, config.nx, config.ny, sigma_w=config.sigma_w, sigma_v=config.sigma_v, n_noise=config.n_noise, A=systems[i]["A"], C=systems[i]["C"])

            repeated_A = np.repeat(sample["A"][np.newaxis,:,:], num_traces, axis=0) #repeat the A matrix for each trace
            sample["A"] = repeated_A #repeat the A matrix for each trace

            repeated_C = np.repeat(sample["C"][np.newaxis,:,:], num_traces, axis=0) #repeat the C matrix for each trace
            sample["C"] = repeated_C #repeat the C matrix for each trace
            samples.extend([{k: v[i] for k, v in sample.items()} for i in range(num_traces)])
            sim_objs.append(fsim)

        with open(output_dirs[M] + f"/data/train_{config.dataset_typ}{config.C_dist}.pkl", "wb") as f:
            pickle.dump(samples, f)

        with open(output_dirs[M] + f"/data/train_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "wb") as f:
            pickle.dump(sim_objs, f)
        

    ## Code to generate validation data below
    # samples = [] #make sure that train and val samples are different
    # sim_objs = [] #make sure that train and val sim_objs are different
    # # print("Generating", num_tasks, "samples for", name)
    # for i in tqdm(range(config.num_val_tasks)):
    #     num_traces = config.num_traces["val"] #number of traces per system must depend on number of systems to have constant total number of traces
    #     fsim, sample = generate_lti_sample(config.C_dist, config.dataset_typ, num_traces, config.n_positions, config.nx, config.ny, sigma_w=config.sigma_w, sigma_v=config.sigma_v, n_noise=config.n_noise, A=systems[i]["A"], C=systems[i]["C"])

    #     repeated_A = np.repeat(sample["A"][np.newaxis,:,:], num_traces, axis=0) #repeat the A matrix for each trace
    #     sample["A"] = repeated_A #repeat the A matrix for each trace

    #     repeated_C = np.repeat(sample["C"][np.newaxis,:,:], num_traces, axis=0) #repeat the C matrix for each trace
    #     sample["C"] = repeated_C #repeat the C matrix for each trace
    #     samples.extend([{k: v[i] for k, v in sample.items()} for i in range(num_traces)])
    #     # raise Exception("just checking fsim type umich_meta_output_predictor/src/collect_data.py")
    #     sim_objs.append(fsim)
    # # print("Saving", len(samples), "samples for", name)

    # with open(output_dir + f"/data/val_{config.dataset_typ}{config.C_dist}.pkl", "wb") as f:
    #     pickle.dump(samples, f)

    # #save fsim to pickle file
    # with open(output_dir + f"/data/val_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "wb") as f:
    #     pickle.dump(sim_objs, f)
    #############################

if __name__ == "__main__":
    config = Config()
    output_dir = "streamlined_mop/ganguli_test"
    collect_data(config, output_dir, Ms=[1, 2, 4], total_traces=40)

    with open("/home/viktor/Documents/Studier/Berkeley/mop_modifications/streamlined_mop/ganguli_test/data/M4_train_gaussA_gauss_C.pkl", "rb") as f:
        data = pickle.load(f)
    
    print(len(data), data[0].keys())
    print(data[0]["A"] == data[9]["A"])
