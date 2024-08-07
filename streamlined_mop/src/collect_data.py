import logging
from dyn_models import generate_lti_sample, generate_lti_sample_new_eig 
from core import Config
from tqdm import tqdm
import pickle
import os
import numpy as np
from models import GPT2

#modify collect data so that it can tolerate multiple traces for one system
def collect_data(model, config, output_dir):

    logger = logging.getLogger(__name__)
    # config = Config()
    config.parse_args()
    print("Collecting data for", config.dataset_typ, config.C_dist)


    # instantiate gpt2 model (FOR THE SAKE OF TESTING, REMOVE LATER)
    # model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
    #              n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)

    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        samples = [] #make sure that train and val samples are different
        sim_objs = [] #make sure that train and val sim_objs are different
        print("Generating", num_tasks, "samples for", name)
        for i in tqdm(range(num_tasks)):
            fsim, sample = generate_lti_sample(config.C_dist, config.dataset_typ, config.num_traces[name], config.n_positions, config.nx, config.ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise)
                    
            repeated_A = np.repeat(sample["A"][np.newaxis,:,:], config.num_traces[name], axis=0) #repeat the A matrix for each trace
            sample["A"] = repeated_A #repeat the A matrix for each trace

            repeated_C = np.repeat(sample["C"][np.newaxis,:,:], config.num_traces[name], axis=0) #repeat the C matrix for each trace
            sample["C"] = repeated_C #repeat the C matrix for each trace
            samples.extend([{k: v[i] for k, v in sample.items()} for i in range(config.num_traces[name])])
            # raise Exception("just checking fsim type umich_meta_output_predictor/src/collect_data.py")
            sim_objs.append(fsim)
        print("Saving", len(samples), "samples for", name)

        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}.pkl", "wb") as f:
            pickle.dump(samples, f)

        #save fsim to pickle file
        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "wb") as f:
            pickle.dump(sim_objs, f)