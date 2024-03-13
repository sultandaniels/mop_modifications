import numpy as np
import scipy as sc
from filterpy.kalman import KalmanFilter


def softplus(x):
    return np.log(1 + np.exp(x))


####################################################################################################
# code that I added
def solve_ricc(A, W):  # solve the Riccati equation for the steady state solution
    L, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    Pi = (V @ (
            (Vinv @ W @ Vinv.T) / (1 - L[:, None] * L)
    ) @ V.T).real
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
            A = np.diag(np.random.uniform(-1, 1, nx)) * 0.95
            A[np.triu_indices(nx, 1)] = np.random.uniform(-1, 1, (nx ** 2 + nx) // 2 - nx)
            self.A = A
        else:
            A = np.random.uniform(-1, 1, (nx, nx))  # fixed the sampling of A to be between -1 and 1
            A /= np.max(np.abs(np.linalg.eigvals(A)))
            self.A = A * 0.95
            # A = np.zeros((nx, nx))
            # A[0,0] = 0.95
            # self.A = A

        self.C = np.eye(nx) if nx == ny else self.construct_C(self.A, ny)

        self.S_state_inf = solve_ricc(self.A, np.eye(nx) * sigma_w ** 2)
        self.sqrt_S_state_inf = np.linalg.cholesky(self.S_state_inf)

        S_state_inf_intermediate = sc.linalg.solve_discrete_are(self.A.T, self.C.T, np.eye(nx) * sigma_w ** 2, np.eye(ny) * sigma_v ** 2)
        self.S_observation_inf = self.C @ S_state_inf_intermediate @ self.C.T + np.eye(ny) * sigma_v ** 2

    # ####################################################################################################

    def simulate(self, traj_len, x0=None):
        ny, nx = self.C.shape
        n_noise = self.n_noise
        xs = [np.random.randn(nx) if x0 is None else x0]  # initial state of dimension nx
        vs = [np.random.randn(ny) * self.sigma_v for _ in range(n_noise)]  # output noise of dimension ny
        ws = [np.random.randn(nx) * self.sigma_w for _ in range(n_noise)]  # state noise of dimension nx
        ys = [self.C @ xs[0] + sum(vs)]  # output of dimension ny
        for _ in range(traj_len):
            x = self.A @ xs[-1] + sum(ws[-n_noise:])
            xs.append(x)
            ws.append(np.random.randn(nx) * self.sigma_w)

            vs.append(np.random.randn(ny) * self.sigma_v)
            y = self.C @ xs[-1] + sum(vs[-n_noise:])
            ys.append(y)
        return np.array(xs).astype("f"), np.array(ys).astype("f")

    ####################################################################################################
    # code that I added

    def simulate_steady(self, batch_size, traj_len):  # change x0 to the steady state distribution
        ny, nx = self.C.shape
        n_noise = self.n_noise
        x0 = np.random.randn(batch_size, nx) @ self.sqrt_S_state_inf.T

        ws = np.random.randn(batch_size, n_noise + traj_len, nx) * self.sigma_w    # state noise of dimension nx
        vs = np.random.randn(batch_size, n_noise + traj_len, ny) * self.sigma_v    # output noise of dimension ny

        xs = [x0]  # initial state of dimension nx
        ys = [xs[0] @ self.C.T + vs[:, :n_noise].sum(axis=1)]  # output of dimension ny

        for i in range(1, traj_len + 1):
            x = xs[-1] @ self.A.T + ws[:, i:i + n_noise].sum(axis=1)
            xs.append(x)

            y = xs[-1] @ self.C.T + vs[:, i:i + n_noise].sum(axis=1)
            ys.append(y)
        return np.stack(xs, axis=1).astype("f"), np.stack(ys, axis=1).astype("f")

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
            if np.linalg.matrix_rank(O) == nx:  # checking if state is observable
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


####################################################################################################
# code that I added
def apply_kf(fsim, ys, sigma_w=None, sigma_v=None, return_obj=False):
    ny, nx = fsim.C.shape

    sigma_w = fsim.sigma_w if sigma_w is None else sigma_w
    sigma_v = fsim.sigma_v if sigma_v is None else sigma_v

    f = KalmanFilter(dim_x=nx, dim_z=ny)
    f.Q = np.eye(nx) * sigma_w ** 2
    f.R = np.eye(ny) * sigma_v ** 2
    f.P = fsim.S_state_inf
    f.x = np.zeros(nx)
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
# code that I added


def _generate_lti_sample(dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri="upperTriA" == dataset_typ, n_noise=n_noise)
    states, obs = fsim.simulate_steady(batch_size, n_positions)
    return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}


def generate_lti_sample(dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    while True:
        fsim, entry = _generate_lti_sample(dataset_typ, batch_size, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise)
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
