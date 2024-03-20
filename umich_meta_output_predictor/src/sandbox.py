import numpy as np
import os
import pickle
import torch


if __name__ == '__main__':
    sims = torch.load('../data/test_sim.pt')
    td = torch.load('../data/test_ypred.pt')

    save_dir = '../data/numpy'
    os.makedirs(save_dir, exist_ok=True)

    params = {
        "F": np.stack([sim.A for sim in sims]),
        "B": np.stack([np.zeros((sim.nx, 0)) for sim in sims]),
        "H": np.stack([sim.C for sim in sims]),
        "sqrt_S_W": np.stack([sim.sigma_w * np.eye(sim.nx) for sim in sims]),
        "sqrt_S_V": np.stack([sim.sigma_v * np.eye(sim.ny) for sim in sims])
    }
    with open(f"{save_dir}/systems.pkl", "wb") as fp:
        pickle.dump(params, fp)

    data = {
        "state": td["states"].numpy(),
        "observation": td["obs"].numpy()
    }
    with open(f"{save_dir}/data.pkl", "wb") as fp:
        pickle.dump(data, fp)




