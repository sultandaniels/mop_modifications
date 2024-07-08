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


def plot_errs(colors, sys, err_lss, err_irreducible, legend_loc="upper right", ax=None, shade=True, normalized=True):
    print("\n\n\nSYS", sys)
    err_rat = np.zeros(2)
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    for i, (name, err_ls) in enumerate(err_lss.items()):
        print("name", name)
        print("err_ls.shape", err_ls.shape)
        if normalized:
            t = np.arange(1, err_ls.shape[-1])
            # if name != "Kalman" and name != "Analytical_Kalman":
            if name == "OLS_ir_length1" or name == "OLS_ir_length2" or name == "OLS_ir_length3" or name == "MOP": 
                normalized_err = (err_ls - err_lss["Kalman"])

                q1, median, q3 = np.quantile(normalized_err[sys], [0.25, 0.5, 0.75], axis=-2)
                scale = median[1]
                q1 = q1/scale
                median = median/scale
                q3 = q3/scale
                handles.extend(ax.plot(t, median[1:], label=name + " sys: " + str(sys), linewidth=3))
                if shade:
                    ax.fill_between(t, q1[1:], q3[1:], facecolor=handles[-1].get_color(), alpha=0.2)
        else:
            if name != "Analytical_Kalman":
                avg, std = err_ls[sys,:,:].mean(axis=(0)), (3/np.sqrt(err_ls.shape[1]))*err_ls[sys,:,:].std(axis=0)
                handles.extend(ax.plot(avg, label=name if name != "OLS_wentinn" else "OLS_ir_length2_unreg", linewidth=3, marker='o' if name == "MOP" else ".", color = colors[i]))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - std, avg + std, facecolor=handles[-1].get_color(), alpha=0.2)
            else:
                handles.extend(ax.plot(err_ls[sys], label=name, linewidth=5, color='#000000'))
            if name == "Kalman":
                err_rat[0] = np.mean(avg)/err_irreducible[sys]
                print("KF (time avergaged mean)/(irreducible): ", err_rat[0])
            if name == "Zero":
                err_rat[1] = np.mean(avg)/err_irreducible[sys]
                print("Zero (time avergaged mean)/(irreducible): ", err_rat[1])
    return handles, err_rat

def plot_errs_conv(ts, colors, sys, err_lss, err_irreducible, train_steps, normalized, legend_loc="upper right", ax=None, shade=True):
    print("\n\n\nSYS", sys)
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    err_avg_t = []
    if not normalized:
        for i, (name, err_ls) in enumerate(err_lss.items()):
            if name == "MOP":
                print("\n\nplotting MOP at step:", train_steps, "\n\n")
                avg, std = err_ls[sys,:,:].mean(axis=(0)), (3/np.sqrt(err_ls.shape[1]))*err_ls[sys,:,:].std(axis=0)
                handles.extend(ax.plot(avg, label=name + train_steps if name != "OLS_wentinn" else "OLS_ir_length2_unreg", linewidth=3, marker='o' if name == "MOP" else "."))#, color = colors[i]))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg -err_irreducible[sys] - std, avg -err_irreducible[sys] + std, facecolor=handles[-1].get_color(), alpha=0.2)

                #set err_avg_t to be the value of avg at the t'th step
                for t in ts:
                    err_avg_t.append(avg[t])

    else: #subtract the irreducible error
        for i, (name, err_ls) in enumerate(err_lss.items()):
            if name == "MOP":
                print("\n\nplotting MOP at step:", train_steps, "\n\n")
                avg, std = err_ls[sys,:,:].mean(axis=(0)), (3/np.sqrt(err_ls.shape[1]))*err_ls[sys,:,:].std(axis=0)
                handles.extend(ax.plot(avg - err_irreducible[sys], label=name + train_steps if name != "OLS_wentinn" else "OLS_ir_length2_unreg", linewidth=3, marker='o' if name == "MOP" else "."))#, color = colors[i]))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - std, avg + std, facecolor=handles[-1].get_color(), alpha=0.2)

                #set err_avg_t to be the value of avg at the t'th step
                for t in ts:
                    err_avg_t.append(avg[t] - err_irreducible[sys])

    return handles, err_avg_t

def spectrum(A, k):
    spec_rad = np.max(np.abs(np.linalg.eigvals(A)))
    return np.linalg.norm(np.linalg.matrix_power(A, k)) / spec_rad ** k

def batch_trace(x: torch.Tensor) -> torch.Tensor:
    # Ensure x has at least two dimensions
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least two dimensions")
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

    def add_data_tensor(self, x, y):
        # Convert self.P to a torch.Tensor
        P_tensor = torch.tensor(self.P, dtype=x.dtype, device=x.device)
        # Perform matrix multiplication
        z = P_tensor @ x / self.lam
        # Before performing the matrix operation, ensure x is a 2-D tensor
        x = x.unsqueeze(0) if x.ndim == 1 else x
        # Ensure x is a 2-D tensor, for example, of shape (10, 1) so that x.mT will be (1, 10)
        x = x.reshape(-1, 1) if x.ndim == 1 else x

        # Ensure z is a 2-D tensor of shape (10, 1) if it's not already
        z = z.reshape(-1, 1) if z.ndim == 1 else z

        # Now perform the matrix multiplication
        result = x.mT @ z
        # Ensure the result of x.T @ z has at least two dimensions
        if result.ndim < 2:
            # Reshape result to have at least two dimensions
            result = result.unsqueeze(0)
            print("shape of result", result.shape)
        alpha = 1 / (1 + batch_trace(result))
        wp = self.mu + y * z
        self.mu = self.mu + z * (y - alpha * batch_trace(x.T @ wp))
        self.P -= alpha * torch.ger(z, z)

class RLS:
    def __init__(self, ni, no, lam=1):
        self.rlss = [RLSSingle(ni, lam) for _ in range(no)]

    def add_data(self, x, y):
        for _y, rls in zip(y, self.rlss):
            rls.add_data(x, _y)

    def add_data_tensor(self, x, y):
        for _y, rls in zip(y, self.rlss):
            rls.add_data_tensor(x, _y)

    def predict(self, x):
        return np.array([rls.mu @ x for rls in self.rlss])
