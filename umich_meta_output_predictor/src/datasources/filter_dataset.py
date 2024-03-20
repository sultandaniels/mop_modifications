from torch.utils.data import Dataset

from core import Config
from dyn_models.filtering_lti import *

config = Config()


class FilterDataset(Dataset):
    def __init__(self, path, use_true_len=False):
        super(FilterDataset, self).__init__()
        with open(path, "rb") as f:
            self.entries = list(torch.load(f).flatten(0, 1))
        self.use_true_len = use_true_len

    def __len__(self):
        return config.train_steps * config.batch_size if not self.use_true_len else len(self.entries)

    def __getitem__(self, idx):
        # generate random entries
        entry = dict(self.entries[idx % len(self.entries)])

        obs = entry.pop("obs")
        L = obs.shape[-2]
        if config.dataset_typ in ["ypred", "noniid", "upperTriA"]:
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
