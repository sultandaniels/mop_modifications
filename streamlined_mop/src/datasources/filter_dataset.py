import pickle

from torch.utils.data import Dataset

from core import Config

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
        # generate random entries
        obs = self.entries[idx % len(self.entries)]["observation"]
        return {"xs": obs[..., :-1, :], "ys": obs[..., 1:, :]}




