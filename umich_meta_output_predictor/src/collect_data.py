import logging
from dyn_models import generate_drone_sample, generate_lti_sample, generate_lti_sample_new_eig  # , generate_pendulum_sample
from core import Config
from tqdm import tqdm
import pickle
import os
import numpy as np

#modify collect data so that it can tolerate multiple traces for one system
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    config = Config()
    config.parse_args()
    print("Collecting data for", config.dataset_typ)
    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        samples = [] #make sure that train and val samples are different
        sim_objs = [] #make sure that train and val sim_objs are different
        print("Generating", num_tasks, "samples for", name)
        for i in tqdm(range(num_tasks)):
            if config.dataset_typ == "drone":
                sim_obj, sample = generate_drone_sample(
                    config.n_positions, sigma_w=1e-1, sigma_v=1e-1, dt=1e-1)
            else:
                fsim, sample = generate_lti_sample(config.C_dist, config.dataset_typ,
                                                   config.num_traces[name],
                                                   config.n_positions,
                                                   config.nx, config.ny,
                                                   sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise)
                    # #save fsim to file
                    # os.makedirs("../data", exist_ok=True)
                    # with open(f"../data/{name}_{config.dataset_typ}_fsim_val.pkl", "wb") as f:
                    #     pickle.dump(fsim, f)
                    
                repeated_A = np.repeat(sample["A"][np.newaxis,:,:], config.num_traces[name], axis=0)
                sample["A"] = repeated_A

                repeated_C = np.repeat(sample["C"][np.newaxis,:,:], config.num_traces[name], axis=0)
                sample["C"] = repeated_C
                # print("shape of sample items:", {k: v.shape for k, v in sample.items()})
                # print("shape of sample A:", sample["A"].shape)
            samples.extend([{k: v[i] for k, v in sample.items()} for i in range(config.num_traces[name])])
            sim_objs.append(fsim)
        print("Saving", len(samples), "samples for", name)
        print("shape of samples:", {k: v.shape for k, v in samples[0].items()})
        os.makedirs("../data", exist_ok=True)
        with open(f"../data/{name}_{config.dataset_typ}_{config.C_dist}.pkl", "wb") as f:
            pickle.dump(samples, f)

        #save fsim to pickle file
        with open(f"../data/{name}_{config.dataset_typ}_{config.C_dist}_sim_objs.pkl", "wb") as f:
            pickle.dump(sim_objs, f)
        
        # os.makedirs("../data", exist_ok=True)
        # with open(f"../data/{name}_sim.pt", "wb") as f:
        #     torch.save(sim_objs, f)
        # with open(f"../data/{name}_{config.dataset_typ}.pt", "wb") as f:
        #     torch.save(torch.stack(samples), f)




