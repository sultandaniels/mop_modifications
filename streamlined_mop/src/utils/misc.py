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
    names = ["MOP", "OLS_ir_1", "OLS_ir_2", "OLS_ir_3", "Analytical_Kalman", "Zero", "Kalman", "OLS_ir_length1_orig", "OLS_ir_length2_orig", "OLS_ir_length3_orig"]
    print("\n\n\nSYS", sys)
    err_rat = np.zeros(2)
    if ax is None:
        fig = plt.figure(figsize=(15, 30))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    color_count = 0
    for i, (name, err_ls) in enumerate(err_lss.items()):
        if name == "Zero":
            print("\n\nSystem", sys, "Zero predictor avged over traces and time:", err_ls[sys,:,:].mean(axis=(0,1)))
        print("name", name)
        print("err_ls.shape", err_ls.shape)
        if name in names: #plot only select curves
            if normalized:
                t = np.arange(1, err_ls.shape[-1])
                
                # if name != "Kalman" and name != "Analytical_Kalman":
                if not (name == "Analytical_Kalman" or name == "Kalman"): 
                    try:
                        normalized_err = (err_ls - err_lss["Kalman"])
                        # #elementwise division of err_ls by err_lss["Analytical_Kalman"]
                        # irr_loses = err_lss["Analytical_Kalman"][:,0]
                        # normalized_err = err_ls[:,:,:]/irr_loses[:,np.newaxis, np.newaxis]
                    except ValueError as e:
                        print("name", name)
                        print("Error: ", e)
                        normalized_err = (err_ls - err_lss["Kalman"].mean(axis=1))

                    q1, median, q3 = np.quantile(normalized_err[sys], [0.45, 0.5, 0.55], axis=-2)
                    # scale = median[1] #scale by the median value at the first time step
                    # q1 = q1/scale
                    # median = median/scale
                    # q3 = q3/scale
                    # if name start "OLS_ir_length" then delete "ir_length" and "orig" from the name
                    if name.startswith("OLS_ir_length"):
                        name = name.replace("ir_length", "")
                        name = name.replace("_orig", "")
                    handles.extend(ax.plot(t, median[1:], marker = None if name.startswith("OLS") else ".", label=name, linewidth=4 if name=="MOP" else 2, linestyle="-." if name.startswith("OLS") else "solid", color = colors[color_count], alpha= 0.7 if name.startswith("OLS") else 1))
                    if shade:
                        ax.fill_between(t, q1[1:], q3[1:], facecolor=handles[-1].get_color(), alpha=0.07)
                    
                    color_count += 1
            else:
                if name in names:
                    if name != "Analytical_Kalman":
                        avg, std = err_ls[sys,:,:].mean(axis=(0)), (3/np.sqrt(err_ls.shape[1]))*err_ls[sys,:,:].std(axis=0)
                        handles.extend(ax.plot(avg, 
                                            label=name if name != "OLS_wentinn" else "OLS_ir_length2_unreg", 
                                            linewidth=1, 
                                            marker='x' if name == "MOP" or name in ["OLS_ir_1", "OLS_ir_2", "OLS_ir_3", "Kalman"] else ".", 
                                            color=colors[color_count], 
                                            markersize=5 if name == "MOP" or name in ["OLS_ir_1", "OLS_ir_2", "OLS_ir_3", "Kalman", "Zero"] else 1))
                        if shade:
                            ax.fill_between(np.arange(err_ls.shape[-1]), avg - std, avg + std, facecolor=handles[-1].get_color(), alpha=0.2)

                        color_count += 1

                    else: #plot the analytical kalman filter
                        handles.extend(ax.plot(err_ls[sys], label=name, linewidth=1, color='#000000'))
                    if name == "Kalman":
                        err_rat[0] = np.mean(avg)/err_irreducible[sys]
                        print("KF (time averaged mean)/(irreducible): ", err_rat[0])
                    if name == "Zero":
                        err_rat[1] = np.mean(avg)/err_irreducible[sys]
                        print("Zero (time averaged mean)/(irreducible): ", err_rat[1])



    return handles, err_rat

def plot_errs_conv(ts, j, colors, sys, err_lss, err_irreducible, train_steps, normalized, legend_loc="upper right", ax=None, shade=True, kal_err=None):
    print("\n\n\nSYS", sys)
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    err_avg_t = []
    err_avg_t_an = []
    for i, (name, err_ls) in enumerate(err_lss.items()):
        if name == "MOP" or name == "MOP_analytical":
            # print("\n\nplotting MOP at step:", train_steps, "\n\n")
            avg, std = err_ls[sys,:,:].mean(axis=(0)), (3/np.sqrt(err_ls.shape[1]))*err_ls[sys,:,:].std(axis=0)

            #compute median and quartiles for the error
            q1, median, q3 = np.quantile(err_ls[sys], [0.45, 0.5, 0.55], axis=-2)
        
            if not normalized:
                print("\nNot Normalized")
                handles.extend(ax.plot(avg, label=name + train_steps if name != "OLS_wentinn" else "OLS_ir_length2_unreg", linewidth=3, marker='o' if name == "MOP" else ".", color = colors[j-1]))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - std, avg + std, facecolor=handles[-1].get_color(), alpha=0.2)

                #set err_avg_t to be the value of avg at the t'th step
                for t in ts:
                    if name == "MOP":
                        # err_avg_t.append((median[t], q1[t], q3[t]))
                        err_avg_t.append((avg[t], avg[t] - std[t], avg[t] + std[t]))
                    elif name == "MOP_analytical":
                        # err_avg_t_an.append((median[t], q1[t], q3[t]))
                        err_avg_t_an.append((avg[t], avg[t] - std[t], avg[t] + std[t]))                    
            else: #subtract the irreducible error
                print("\nNormalized")
                handles.extend(ax.plot(avg - err_irreducible[sys], label=name + train_steps if name != "OLS_wentinn" else "OLS_ir_length2_unreg", linewidth=3, marker='o' if name == "MOP" else ".", color = colors[j-1]))
                if shade:
                    ax.fill_between(np.arange(err_ls.shape[-1]), avg - err_irreducible[sys] - std, avg - err_irreducible[sys] + std, facecolor=handles[-1].get_color(), alpha=0.2)

                #set err_avg_t to be the value of avg at the t'th step
                for t in ts:
                    if name == "MOP":
                        # err_avg_t.append((median[t] - kal_err[sys][t], q1[t] - kal_err[sys][t], q3[t] - kal_err[sys][t]))
                        err_avg_t.append((avg[t] - kal_err[sys][t], avg[t] - std[t] - kal_err[sys][t], avg[t] + std[t] - kal_err[sys][t]))
                    elif name == "MOP_analytical":
                        # err_avg_t_an.append((median[t] - kal_err[sys][t], q1[t] - kal_err[sys][t], q3[t] - kal_err[sys][t]))
                        err_avg_t_an.append((avg[t] - kal_err[sys][t], avg[t] - std[t] - kal_err[sys][t], avg[t] + std[t] - kal_err[sys][t]))
    return handles, err_avg_t, err_avg_t_an

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
        self.P = np.eye(ni+5)
        self.mu = np.zeros(ni+5)

    def add_data(self, x, y):
        z = self.P @ x / self.lam
        alpha = 1 / (1 + x.T @ z)
        wp = self.mu + y * z
        self.mu = self.mu + z * (y - alpha * x.T @ wp)
        #print("self.mu.shape:", self.mu.shape)
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
        #print("shape of x:", x.shape)
        return np.array([rls.mu @ x for rls in self.rlss])
