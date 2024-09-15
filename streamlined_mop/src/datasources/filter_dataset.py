from torch.utils.data import Dataset
from dyn_models.filtering_lti import *
from core import Config
import torch
import pickle

config = Config()


class FilterDataset(Dataset):
    def __init__(self, path, use_true_len=False):
        super(FilterDataset, self).__init__()
        self.load(path)
        self.use_true_len = use_true_len

    def load(self, path):
        with open(path, "rb") as f:
            self.entries = pickle.load(f)

    def __len__(self):
        return config.train_steps * config.batch_size if not self.use_true_len else len(self.entries)

    def __getitem__(self, idx):
        # generate random entites
        entry = self.entries[idx % len(self.entries)].copy()

        obs = entry.pop("obs")
        L = obs.shape[-2]
        if config.dataset_typ in ["unifA", "noniid", "upperTriA", "rotDiagA", "gaussA", "gaussA_noscale", "single_system", "cond_num"]:
            entry["xs"] = np.take(obs, np.arange(L - 1), axis=-2)
            entry["ys"] = np.take(obs, np.arange(1, L), axis=-2)
        elif config.dataset_typ == "drone":
            actions = entry.pop("actions")
            entry["xs"] = np.concatenate([np.take(obs, np.arange(L - 1), axis=-2), actions], axis=-1)
            entry["ys"] = np.take(obs, np.arange(1, L), axis=-2)
        else:
            raise NotImplementedError(f"{config.dataset_typ} is not implemented")

        torch_entry = dict([
            (k, (torch.from_numpy(a) if isinstance(a, np.ndarray) else a).to(torch.float32))
            for k, a in entry.items()])
        return torch_entry
