import logging
import os

import torch
from tqdm import tqdm

from core import Config
from dyn_models import generate_drone_sample, generate_lti_sample  # , generate_pendulum_sample

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    config = Config()
    config.parse_args()
    print("Collecting data for", config.dataset_typ)
    for name, num_tasks in config.num_tasks.items():
        sim_objs, samples = [], []
        print("Generating", num_tasks, "samples for", name)
        for i in tqdm(range(num_tasks)):
            if config.dataset_typ == "drone":
                sim_obj, sample = generate_drone_sample(
                    config.n_positions, sigma_w=1e-1, sigma_v=1e-1, dt=1e-1)
            else:
                sim_obj, sample = generate_lti_sample(config.dataset_typ,
                                                      config.num_traces[name],
                                                      config.n_positions,
                                                      config.nx, config.ny,
                                                      sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise)
            sim_objs.append(sim_obj)
            samples.append(sample)

        os.makedirs("../data", exist_ok=True)
        with open(f"../data/{name}_sim.pt", "wb") as f:
            torch.save(sim_objs, f)
        with open(f"../data/{name}_{config.dataset_typ}.pt", "wb") as f:
            torch.save(torch.stack(samples), f)
