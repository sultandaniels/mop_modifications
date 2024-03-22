import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def log_info(st):
    if type(st) != str:
        st = str(st)
    return logger.info(st)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def set_seed(seed=0, fully_reproducible=False):
    # Improve reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if fully_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def plot_errs(sys, err_lss, err_irreducible, legend_loc="upper right", ax=None, shade=True, normalized=True):
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    for i, (name, err_ls) in enumerate(err_lss.items()):

        print("name", name)
        print("err_ls.shape", err_ls.shape)
        # if name != "Analytical_Kalman":
        #     traj_errs = err_ls.sum(axis=-1)
        #     print("traj_errs.shape", traj_errs.shape)
        #     print(name, "{:.2f}".format(traj_errs.mean(axis=(0, 1))))
        # else:
        #     print(name, "{:.2f}".format(err_ls[0]))
        if normalized:
            t = np.arange(1, err_ls.shape[-1])
            if name != "Kalman":
                normalized_err = (err_ls - err_lss["Kalman"]) / np.expand_dims(err_irreducible, axis=tuple(range(1, err_ls.ndim)))

                q1, median, q3 = np.quantile(normalized_err, [0.25, 0.5, 0.75], axis=-2).mean(axis=1)
                handles.extend(ax.plot(t, median[1:], label=name, linewidth=3))
                if shade:
                    ax.fill_between(t, q1[1:], q3[1:], facecolor=handles[-1].get_color(), alpha=0.2)
        else:
            if name != "Analytical_Kalman":
                avg, std = err_ls[sys,:,:].mean(axis=(0)), (3/np.sqrt(err_ls.shape[1]))*err_ls[sys,:,:].std(axis=0)
                handles.extend(ax.plot(avg, label=name if name != "OLS_wentinn" else "OLS_ir_length2_unreg", linewidth=3, marker='o' if name == "MOP" else "."))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - std, avg + std, facecolor=handles[-1].get_color(), alpha=0.2)
            else:
                handles.extend(ax.plot(err_ls[sys], label=name, linewidth=3))
    return handles


def spectrum(A, k):
    spec_rad = np.max(np.abs(np.linalg.eigvals(A)))
    return np.linalg.norm(np.linalg.matrix_power(A, k)) / spec_rad ** k


def batch_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


class RLSSingle:
    def __init__(self, ni, lam=1):
        self.lam = lam
        self.P = np.eye(ni)
        self.mu = np.zeros(ni)

    def add_data(self, x, y):
        z = self.P @ x / self.lam
        alpha = 1 / (1 + x.T @ z)
        wp = self.mu + y * z
        self.mu = self.mu + z * (y - alpha * x.T @ wp)
        self.P -= alpha * np.outer(z, z)


class RLS:
    def __init__(self, ni, no, lam=1):
        self.rlss = [RLSSingle(ni, lam) for _ in range(no)]

    def add_data(self, x, y):
        for _y, rls in zip(y, self.rlss):
            rls.add_data(x, _y)

    def predict(self, x):
        return np.array([rls.mu @ x for rls in self.rlss])
