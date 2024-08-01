import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def plot_errs(colors, sys, err_lss, err_irreducible, legend_loc="upper right", ax=None, shade=True, normalized=False):
    print("\n\n\nSYS", sys)
    err_rat = np.zeros(2)
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    for i, (name, err_ls) in enumerate(err_lss.items()):
        err_ls = err_ls.cpu().numpy()
        print("name", name)
        print("err_ls.shape", err_ls.shape)
        if normalized:
            t = np.arange(1, err_ls.shape[-1])
            # if name != "Kalman" and name != "Analytical_Kalman":
            if name == "OLS_ir_length1" or name == "OLS_ir_length2" or name == "OLS_ir_length3" or name == "MOP":
                normalized_err = (err_ls - err_lss["Kalman"])

                q1, median, q3 = np.quantile(normalized_err[sys], [0.25, 0.5, 0.75], axis=-2)
                scale = median[1]
                q1 = q1 / scale
                median = median / scale
                q3 = q3 / scale
                handles.extend(ax.plot(t, median[1:], label=name + " sys: " + str(sys), linewidth=3))
                if shade:
                    ax.fill_between(t, q1[1:], q3[1:], facecolor=handles[-1].get_color(), alpha=0.2)
        else:
            if name != "Analytical_Kalman":
                avg, std = err_ls[sys, :, :].mean(axis=0), (3 / np.sqrt(err_ls.shape[1])) * err_ls[sys, :, :].std(
                    axis=0)
                handles.extend(ax.plot(avg,
                                       label=name if name != "OLS_wentinn" else "OLS_ir_length2_unreg",
                                       linewidth=1,
                                       marker='x' if name == "MOP" or name in ["OLS_ir_1", "OLS_ir_2", "OLS_ir_3",
                                                                               "Kalman"] else ".",
                                       color=colors[i],
                                       markersize=5 if name == "MOP" or name in ["OLS_ir_1", "OLS_ir_2", "OLS_ir_3",
                                                                                 "Kalman", "Zero"] else 1))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - std, avg + std,
                                    facecolor=handles[-1].get_color(), alpha=0.2)
            else:  # plot the analytical kalman filter
                handles.extend(ax.plot(err_ls[sys], label=name, linewidth=2, color='#000000'))
                # handles.extend(ax.plot(err_irreducible[sys], label=name, linewidth=2, color='#000000'))
            if name == "Kalman":
                err_rat[0] = np.mean(avg) / err_irreducible[sys]
                print("KF (time avergaged mean)/(irreducible): ", err_rat[0])
            if name == "Zero":
                err_rat[1] = np.mean(avg) / err_irreducible[sys]
                print("Zero (time avergaged mean)/(irreducible): ", err_rat[1])
    return handles, err_rat


def plot_errs_conv(ts, j, colors, sys, err_lss, err_irreducible, train_steps, normalized, legend_loc="upper right",
                   ax=None, shade=True):
    print("\n\n\nSYS", sys)
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    err_avg_t = []
    for i, (name, err_ls) in enumerate(err_lss.items()):
        if name == "MOP":
            print("\n\nplotting MOP at step:", train_steps, "\n\n")
            avg, std = err_ls[sys, :, :].mean(axis=(0)), (3 / np.sqrt(err_ls.shape[1])) * err_ls[sys, :, :].std(axis=0)

            if not normalized:
                # compute median and quartiles for the error
                q1, median, q3 = np.quantile(err_ls[sys], [0.25, 0.5, 0.75], axis=-2)
                print("q1.shape", q1.shape)
                print("median.shape", median.shape)
                print("q3.shape", q3.shape)

                handles.extend(
                    ax.plot(avg, label=name + train_steps if name != "OLS_wentinn" else "OLS_ir_length2_unreg",
                            linewidth=3, marker='o' if name == "MOP" else ".", color=colors[j - 1]))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - std, avg + std,
                                    facecolor=handles[-1].get_color(), alpha=0.2)

                # set err_avg_t to be the value of avg at the t'th step
                for t in ts:
                    err_avg_t.append((median[t], q1[t], q3[t]))

            else:  # subtract the irreducible error
                # compute median and quartiles for the error
                q1, median, q3 = np.quantile(err_ls[sys], [0.25, 0.5, 0.75], axis=-2)

                handles.extend(ax.plot(avg - err_irreducible[sys],
                                       label=name + train_steps if name != "OLS_wentinn" else "OLS_ir_length2_unreg",
                                       linewidth=3, marker='o' if name == "MOP" else ".", color=colors[j - 1]))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - err_irreducible[sys] - std,
                                    avg - err_irreducible[sys] + std, facecolor=handles[-1].get_color(), alpha=0.2)

                # set err_avg_t to be the value of avg at the t'th step
                for t in ts:
                    err_avg_t.append(
                        (median[t] - err_irreducible[sys], q1[t] - err_irreducible[sys], q3[t] - err_irreducible[sys]))
    return handles, err_avg_t


def spectrum(A, k):
    spec_rad = np.max(np.abs(np.linalg.eigvals(A)))
    return np.linalg.norm(np.linalg.matrix_power(A, k)) / spec_rad ** k




