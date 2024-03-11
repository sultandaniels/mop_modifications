import numpy as np
import sympy as sym
from sympy import *
from sympy.solvers.solveset import linsolve
from filterpy.kalman import KalmanFilter
import torch


def softplus(x):
    return np.log(1 + np.exp(x))

####################################################################################################
#code that I added
def solve_ricc(A, W): #solve the Riccati equation for the steady state solution
    A = torch.from_numpy(A) #convert A to a tensor
    W = torch.from_numpy(W).to(torch.complex128) #convert W to a complex tensor
    L, V = torch.linalg.eig(A)
    Vinv = torch.linalg.inv(V)
    Pi = (V @ (
        (((L[:, None] ** -1) * Vinv) @ W @ Vinv.T) / (1 / L[:, None] - L)
    ) @ V.T).real
    return Pi

def solve_ricc_sym(A, W): #solve the Riccati equation for the steady state solution
    n = A.shape[0]
    symbols = [sym.var(f'm_{i+1}{j+1}') for i in range(n) for j in range(n)]
    # Create the matrix M using the Matrix function and the symbols list
    M = sym.Matrix(n, n, symbols)
    # left multiply M by A and right multiply that product by A.T
    mat = A @ M @ A.T
    # subtract mat from M and and subtract W
    Ricc = M - mat - W
    eqs = []
    for i in range(n):
        for j in range(n):
            eqs.append(Ricc[i,j])

    flat_pi = linsolve(eqs, symbols)
    Pi = np.array(flat_pi.args).reshape((n,n))
    return Pi
####################################################################################################


class FilterSim:
    # def __init__(self, nx=3, ny=2, sigma_w=1e-1, sigma_v=1e-1, tri=False, n_noise=1):
    #     self.sigma_w = sigma_w
    #     self.sigma_v = sigma_v

    #     self.n_noise = n_noise

    #     if tri:
    #         A = np.diag(np.random.rand(nx) * 2 - 1) * 0.95
    #         A[np.triu_indices(nx, 1)] = np.random.rand((nx ** 2 + nx) // 2 - nx) * 2 - 1
    #         self.A = A
    #     else:
    #         A = np.random.rand(nx, nx)
    #         A /= np.max(np.abs(np.linalg.eigvals(A)))
    #         self.A = A * 0.95

    #     self.C = np.eye(nx) if nx == ny else self.construct_C(self.A, ny)

    # ####################################################################################################
    # #code that I added
    def __init__(self, nx=3, ny=2, sigma_w=1e-1, sigma_v=1e-1, tri=False, n_noise=1):
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v

        self.n_noise = n_noise

        if tri:
            A = np.diag(np.random.rand(nx) * 2 - 1) * 0.95
            A[np.triu_indices(nx, 1)] = np.random.rand((nx ** 2 + nx) // 2 - nx) * 2 - 1
            self.A = A
        else:
            A = 2*np.random.rand(nx, nx) - 1 #fixed the sampling of A to be between -1 and 1
            A /= np.max(np.abs(np.linalg.eigvals(A)))
            self.A = A * 0.95
            # A = np.zeros((nx, nx))
            # A[0,0] = 0.95
            # self.A = A

        self.C = np.eye(nx) if nx == ny else self.construct_C(self.A, ny)
    # ####################################################################################################

    def simulate(self, traj_len, x0=None):
        ny, nx = self.C.shape
        n_noise = self.n_noise
        xs = [np.random.randn(nx) if x0 is None else x0] # initial state of dimension nx
        vs = [np.random.randn(ny) * self.sigma_v for _ in range(n_noise)] # output noise of dimension ny
        ws = [np.random.randn(nx) * self.sigma_w for _ in range(n_noise)] # state noise of dimension nx
        ys = [self.C @ xs[0] + sum(vs)] # output of dimension ny
        for _ in range(traj_len):
            x = self.A @ xs[-1] + sum(ws[-n_noise:])
            xs.append(x)
            ws.append(np.random.randn(nx) * self.sigma_w)

            vs.append(np.random.randn(ny) * self.sigma_v)
            y = self.C @ xs[-1] + sum(vs[-n_noise:])
            ys.append(y)
        return np.array(xs).astype("f"), np.array(ys).astype("f")
    
    ####################################################################################################
    #code that I added

    def simulate_steady(self, traj_len, x0):#change x0 to the steady state distribution
            ny, nx = self.C.shape
            n_noise = self.n_noise
            xs = [x0] # initial state of dimension nx
            vs = [np.random.randn(ny) * self.sigma_v for _ in range(n_noise)] # output noise of dimension ny
            ws = [np.random.randn(nx) * self.sigma_w for _ in range(n_noise)] # state noise of dimension nx
            ys = [self.C @ xs[0] + sum(vs)] # output of dimension ny
            for _ in range(traj_len):
                x = self.A @ xs[-1] + sum(ws[-n_noise:])
                xs.append(x)
                ws.append(np.random.randn(nx) * self.sigma_w)

                vs.append(np.random.randn(ny) * self.sigma_v)
                y = self.C @ xs[-1] + sum(vs[-n_noise:])
                ys.append(y)
            return np.array(xs).astype("f"), np.array(ys).astype("f")
    
    def simulate_steady_new(self, traj_len, x0):#change x0 to the steady state distribution
            ny, nx = self.C.shape
            n_noise = self.n_noise
            xs = [x0] # initial state of dimension nx
            vs = np.random.randn(traj_len, ny) * self.sigma_v # output noise of dimension ny
            ws = np.random.randn(traj_len + 1, nx) * self.sigma_w # state noise of dimension nx
            for i in range(traj_len):
                x = self.A @ xs[-1] + ws[i]
            xs = np.stack(xs).astype("f")
            ys = (xs @ self.C.T + vs).astype("f")
            return xs, ys


    ####################################################################################################
    


    @staticmethod
    def construct_C(A, ny):
        nx = A.shape[0]
        _O = [np.eye(nx)]
        for _ in range(nx - 1):
            _O.append(_O[-1] @ A)
        while True:
            C = np.random.rand(ny, nx)
            O = np.concatenate([C @ o for o in _O], axis=0)
            if np.linalg.matrix_rank(O) == nx: #checking if state is observable
                break
        return C.astype("f")

    # ####################################################################################################
    # #code that I added
    # @staticmethod
    # def construct_C(A, ny):
    #     nx = A.shape[0]
    #     C = np.random.rand(ny, nx)
    #     return C.astype("f")
    # ####################################################################################################


def apply_kf_du(fsim, ys, x0=None, P0=None, sigma_w=None, sigma_v=None, return_obj=False): #errors  in P0 initialization, check if f.x is xhat or the actual state, check what f.predict does
    ny, nx = fsim.C.shape

    sigma_w = fsim.sigma_w if sigma_w is None else sigma_w
    sigma_v = fsim.sigma_v if sigma_v is None else sigma_v

    f = KalmanFilter(dim_x=nx, dim_z=ny)
    f.Q = np.eye(nx) * sigma_w ** 2
    f.R = np.eye(ny) * sigma_v ** 2
    f.P = np.eye(nx) if P0 is None else P0
    f.x = np.zeros(nx) if x0 is None else x0
    f.F = fsim.A
    f.H = fsim.C

    ls = [fsim.C @ f.x]
    for y in ys:
        f.update(y)
        f.predict()
        ls.append(fsim.C @ f.x)
    ls = np.array(ls)
    return (f, ls) if return_obj else ls

####################################################################################################
#code that I added
def apply_kf_steady(fsim, ys, x0, P0, sigma_w=None, sigma_v=None, return_obj=False): 
    ny, nx = fsim.C.shape

    sigma_w = fsim.sigma_w if sigma_w is None else sigma_w
    sigma_v = fsim.sigma_v if sigma_v is None else sigma_v

    f = KalmanFilter(dim_x=nx, dim_z=ny)
    f.Q = np.eye(nx) * sigma_w ** 2
    f.R = np.eye(ny) * sigma_v ** 2
    f.P = np.eye(nx) if P0 is None else P0
    f.x = np.zeros(nx) if x0 is None else x0
    f.F = fsim.A
    f.H = fsim.C

    ls = [fsim.C @ f.x]
    for y in ys:
        f.predict()
        ls.append(fsim.C @ f.x)
        f.update(y) #swapped this line from line 202
    ls = np.array(ls)
    return (f, ls) if return_obj else ls
####################################################################################################


def _generate_lti_sample(dataset_typ, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri="upperTriA" == dataset_typ, n_noise=n_noise)
    states, obs = fsim.simulate(n_positions)
    return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}


def generate_lti_sample(dataset_typ, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    while True:
        fsim, entry = _generate_lti_sample(dataset_typ, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise)
        if check_validity(entry):
            return fsim, entry
        

####################################################################################################
#code that I added

def _generate_lti_sample_steady(dataset_typ, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri="upperTriA" == dataset_typ, n_noise=n_noise)
    Pi = solve_ricc(fsim.A, (sigma_w**2)*np.eye(nx)) #set the initial state covariance matrix Pi to the steady state solution of the Riccati equation
    x0_steady = np.random.multivariate_normal(np.zeros(nx), Pi) #set the initial state to the steady state distribution
    # print("x0_steady squared", np.square(x0_steady))
    states, obs = fsim.simulate_steady(n_positions, x0_steady)
    return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}


def generate_lti_sample_steady(dataset_typ, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    while True:
        fsim, entry = _generate_lti_sample_steady(dataset_typ, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise)
        if check_validity(entry):
            return fsim, entry
        
####################################################################################################


def generate_changing_lti_sample(n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim1 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise)
    fsim2 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise)

    _xs, _ys = fsim1.simulate(n_positions)
    while not check_validity({"states": _xs, "obs": _ys}):
        _xs, _ys = fsim1.simulate(n_positions)
    _xs_cont, _ys_cont = fsim2.simulate(n_positions, x0=_xs[-1])
    while not check_validity({"states": _xs_cont, "obs": _ys_cont}):
        _xs_cont, _ys_cont = fsim2.simulate(n_positions, x0=_xs[-1])
    y_seq = np.concatenate([_ys[:-1], _ys_cont], axis=0)
    return fsim1, {"obs": y_seq}


def check_validity(entry):
    if entry is None:
        return False
    states, obs = entry["states"], entry["obs"]
    return np.max(np.abs(states)) < 50 and np.max(np.abs(obs)) < 50


