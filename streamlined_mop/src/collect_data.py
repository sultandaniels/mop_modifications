import logging
from dyn_models import generate_lti_sample, generate_lti_sample_new_eig 
from core import Config
from tqdm import tqdm
import pickle
import os
import numpy as np
from models import GPT2
import argparse

#modify collect data so that it can tolerate multiple traces for one system
def collect_data(model, config, output_dir, only="", train_mix=False):

    logger = logging.getLogger(__name__)
    # config = Config()
    print("Collecting data for", config.dataset_typ, config.C_dist)


    # instantiate gpt2 model (FOR THE SAKE OF TESTING, REMOVE LATER)
    # model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
    #              n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)

    zero_count = 0
    one_count = 0
    two_count = 0
    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        if only and name != only: #if only is specified, skip the other dataset
            continue
        samples = [] #make sure that train and val samples are different
        sim_objs = [] #make sure that train and val sim_objs are different
        print("Generating", num_tasks, "samples for", name)

        if name == "train" and train_mix:
            A_dists = ["gaussA", "upperTriA", "rotDiagA"]

        for i in tqdm(range(num_tasks)):
            if name == "train" and train_mix:
                if int(np.floor(len(A_dists)*i/num_tasks)) > 2:
                    index = 0
                else:
                    index = int(np.floor(len(A_dists)*i/num_tasks))

                if index == 0:
                    zero_count += 1
                elif index == 1:
                    one_count += 1
                elif index == 2:
                    two_count += 1

                config.override("dataset_typ", A_dists[index]) #override the dataset_typ

            fsim, sample = generate_lti_sample(config.C_dist, config.dataset_typ, config.num_traces[name], config.n_positions, config.nx, config.ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise)
                    
            repeated_A = np.repeat(sample["A"][np.newaxis,:,:], config.num_traces[name], axis=0) #repeat the A matrix for each trace
            sample["A"] = repeated_A #repeat the A matrix for each trace

            repeated_C = np.repeat(sample["C"][np.newaxis,:,:], config.num_traces[name], axis=0) #repeat the C matrix for each trace
            sample["C"] = repeated_C #repeat the C matrix for each trace
            samples.extend([{k: v[i] for k, v in sample.items()} for i in range(config.num_traces[name])])
            # raise Exception("just checking fsim type umich_meta_output_predictor/src/collect_data.py")
            sim_objs.append(fsim)
        print("Saving", len(samples), "samples for", name)

        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}" + "_mix" if train_mix and name == "train" else "" + ".pkl", "wb") as f:
            pickle.dump(samples, f)

        #save fsim to pickle file
        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}" + "_mix" if train_mix and name == "train" else "" + "_sim_objs.pkl", "wb") as f:
            pickle.dump(sim_objs, f)

    if train_mix:
        print("zero_count:", zero_count)
        print("one_count:", one_count)
        print("two_count:", two_count)
        print("config.dataset_typ:", config.dataset_typ)

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='What data to collect.')

    # Add the arguments
    parser.add_argument('--val', help='Boolean. only generate validation data', action='store_true')
    parser.add_argument('--train', help='Boolean. only generate training data', action='store_true')
    parser.add_argument('--train_mix', help='Boolean. generate training data from gaussian, uppertriA, and rotdiagA', action='store_true')


    # Parse the arguments
    args = parser.parse_args()
    print("only val:", args.val)
    print("only train:", args.train)
    print("train_mix:", args.train_mix)

    train_mix = args.train_mix

    # Now you can use the flag
    if args.val:
        only = "val"
    elif args.train:
        only = "train"
    else:
        only = ""

    config = Config()
    model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                 n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
    
    collect_data(model, config, "../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C", only, train_mix)