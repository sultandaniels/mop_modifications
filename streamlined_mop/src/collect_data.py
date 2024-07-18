import pickle
from argparse import Namespace

import torch
from tensordict import TensorDict
from tqdm import tqdm

from dyn_models import generate_lti_system
from system.linear_time_invariant import LTISystem


# modify collect data so that it can tolerate multiple traces for one system
def collect_data(config, output_dir):
    config.parse_args()
    print("Collecting data for", config.dataset_typ, config.C_dist)
    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        _sim_objs = []  # make sure that train and val sim_objs are different
        print("Generating", num_tasks, "samples for", name)
        for _ in tqdm(range(num_tasks)):
            _sim_objs.append(generate_lti_system(config.C_dist, config.dataset_typ, config.nx, config.ny, sigma_w=1e-1, sigma_v=1e-1,
                                                 n_noise=config.n_noise))

        problem_shape = Namespace(
            environment=Namespace(observation=config.ny),
            controller=Namespace()
        )
        sys_td = torch.stack([
            TensorDict.from_dict({"environment": {
                "F": torch.Tensor(sim_obj.A),
                "B": {},
                "H": torch.Tensor(sim_obj.C),
                "sqrt_S_W": sim_obj.sigma_w * torch.eye(config.nx),
                "sqrt_S_V": sim_obj.sigma_v * torch.eye(config.ny)
            }}, batch_size=())
            for sim_obj in _sim_objs
        ], dim=0)

        with torch.set_grad_enabled(False):
            lsg = LTISystem(problem_shape, sys_td)
            samples = lsg.generate_dataset(config.num_traces[name], config.n_positions + 1).reshape(-1, config.n_positions + 1)["environment"]
        print("Saving", len(samples), "samples for", name)

        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}.pkl", "wb") as f:
            pickle.dump(samples, f)

        # save fsim to pickle file
        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "wb") as f:
            pickle.dump(lsg, f)




