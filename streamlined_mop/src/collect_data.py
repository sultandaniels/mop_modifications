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
    for name, num_tasks in zip(["train", "val", "val_noiseless"], [config.num_tasks, config.num_val_tasks, config.num_val_tasks]):
        if name == "val_noiseless":
            noise_value = 0
        else:
            noise_value = 1e-1

        _sim_objs = []  # make sure that train and val sim_objs are different
        print("Generating", num_tasks, "samples for", name)
        for _ in tqdm(range(num_tasks)):
            _sim_objs.append(generate_lti_system(config.C_dist, config.dataset_typ, config.nx, config.ny, sigma_w=noise_value, sigma_v=noise_value,
                                                 n_noise=config.n_noise))

        problem_shape = Namespace(
            environment=Namespace(observation=config.ny),
            controller=Namespace()
        )

        print("sim_obj.sigma_w", _sim_objs[0].sigma_w)
        print("sim_obj.sigma_v", _sim_objs[0].sigma_v)
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

        if name == "train":
            trace_num = config.num_traces[name]
        else:
            trace_num = config.num_traces["val"]

        with torch.set_grad_enabled(False):
            lsg = LTISystem(problem_shape, sys_td)
            samples = lsg.generate_dataset(trace_num, config.n_positions + 1).reshape(-1, config.n_positions + 1)["environment"]
        print("Saving", len(samples), "samples for", name)

        # check if samples contains nan
        if torch.isnan(samples).any():
            raise ValueError("samples contain nan in collect_data.py")

        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}.pkl", "wb") as f:
            pickle.dump(samples, f)

        # save fsim to pickle file
        with open(output_dir + f"/data/{name}_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "wb") as f:
            pickle.dump(lsg, f)




