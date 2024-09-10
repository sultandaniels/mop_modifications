import numpy as np
import scipy as sc
from numpy import linalg as lin
from scipy import linalg as la
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

def gen_A(elow, ehigh, n): # generates a 2d A matrix with evalue magnitudes in between elow and ehigh. For matrices larger than 2d, when there are complex evalues it just ensures that all evalues are below ehigh and at least one evalue is above elow.
    if elow > ehigh:
        raise ValueError("elow must be less than ehigh")

    # bound = np.random.uniform(elow, ehigh, size=(2,)) #sample a random bound for the evalues
    # #sort bound in descending order
    # bound = np.sort(bound)[::-1]
    a = ehigh #bound[0]
    b = elow #bound[1]

    # mat = np.random.normal(loc=0, scale = 1, size=(n,n)) #produce random square matrix with normal distribution
    mat = np.random.uniform(-1, 1, (n,n)) #produce random square matrix with uniform distribution

    # get eigenvalues of mat
    eigs = lin.eigvals(mat)
    # sort eignvalues by magnitude in descending order
    sorted_indices = np.argsort(np.abs(eigs))[::-1] #changed from np.sort(eigs)[::-1]
    eigs = eigs[sorted_indices] #sort the evalues

    eps = 1e-10 #small number to check if evalues are out of bounds
    if np.iscomplex(eigs).any(): #if there are complex evalues
        # scale the eigenvalues to the bound
        alpha = a #a + 0.5*(b-a) #number to scale the evalues to
        out = (alpha/np.abs(eigs[0]))*mat #scale the matrix

        if (np.sum(np.abs(lin.eigvals(out)) > elow - eps) < 1) or np.any(np.abs(lin.eigvals(out)) > ehigh+eps): #if there are no evalues above elow or there are evalues above ehigh
            print("np.abs(lin.eigvals(out)):", np.abs(lin.eigvals(out)))
            raise ValueError("evalues out of bounds (complex)")
    
    else: #if there are no complex evalues
        print("all real evalues")
        A = ((b-a)/(eigs[0] - eigs[-1]))*(mat - eigs[-1]*np.eye(n)) + a*np.eye(n) #subtract the smallest evalue from the matrix
        T,Z = la.schur(A) #get the schur decomposition
        sgn = 2*np.random.randint(2, size=n) - np.ones((n,)) #randomly choose a sign for each evalue
        for i in range(n):
            T[i,i] = T[i,i]*sgn[i] #multiply the diagonal by the sign
        out = Z@T@Z.T #reconstruct the matrix
        if (np.any(np.abs(lin.eigvals(out)) < elow-eps) or np.any(np.abs(lin.eigvals(out)) > ehigh+eps)): #if there are evalues below elow or above ehigh
            print("np.abs(lin.eigvals(out)):", np.abs(lin.eigvals(out)))
            raise ValueError("evalues out of bounds (real)")

    print("np.abs(lin.eigvals(out)):", np.abs(lin.eigvals(out))) 
    return out


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
    def __init__(self, nx, ny, sigma_w, sigma_v, tri, C_dist, n_noise, new_eig):
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v

        self.n_noise = n_noise

        if tri == "upperTriA":
            A = np.diag(np.random.uniform(-1, 1, nx)) * 0.95
            A[np.triu_indices(nx, 1)] = np.random.uniform(-1, 1, (nx ** 2 + nx) // 2 - nx)
            self.A = A
        elif tri == "rotDiagA":
            A = np.diag([0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9]) * 0.95 #generate a random diagonal matrix
            # Generate a random 10x10 matrix
            random_matrix = np.random.randn(10, 10)

            # Use QR decomposition to get a random rotation matrix
            Q, R = np.linalg.qr(random_matrix)
                         
            self.A = Q @ A @ Q.T 
        elif tri == "gaussA":
            A = np.sqrt(0.33)*np.random.randn(nx, nx) #same second moment as uniform(-1,1)
            A /= np.max(np.abs(np.linalg.eigvals(A)))
            self.A = A * 0.95 #scale the matrix
        elif tri == "gaussA_noscale":
            self.A = np.sqrt(0.33)*np.random.randn(nx, nx) #same second moment as uniform(-1,1)
        elif tri == "single_system":
            A = np.array([
                [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152],
                [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293, 0.0202184, 0.83261985, 0.77815675, 0.87001215],
                [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443, 0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194],
                [0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898, 0.6176355, 0.61209572, 0.616934, 0.94374808, 0.6818203],
                [0.3595079, 0.43703195, 0.6976312, 0.06022547, 0.66676672, 0.67063787, 0.21038256, 0.1289263, 0.31542835, 0.36371077],
                [0.57019677, 0.43860151, 0.98837384, 0.10204481, 0.20887676, 0.16130952, 0.65310833, 0.2532916, 0.46631077, 0.24442559],
                [0.15896958, 0.11037514, 0.65632959, 0.13818295, 0.19658236, 0.36872517, 0.82099323, 0.09710128, 0.83794491, 0.09609841],
                [0.97645947, 0.4686512, 0.97676109, 0.60484552, 0.73926358, 0.03918779, 0.28280696, 0.12019656, 0.2961402, 0.11872772],
                [0.31798318, 0.41426299, 0.0641475, 0.69247212, 0.56660145, 0.26538949, 0.52324805, 0.09394051, 0.5759465, 0.9292962],
                [0.31856895, 0.66741038, 0.13179786, 0.7163272, 0.28940609, 0.18319136, 0.58651293, 0.02010755, 0.82894003, 0.00469548]
            ]) #generate a random specific matrix
            A /= np.max(np.abs(np.linalg.eigvals(A)))
            self.A = A * 0.95 #scale the matrix
        else:
            if new_eig:
                self.A = gen_A(0.97, 0.99, nx)
            else:
                A = np.random.uniform(-1, 1, (nx, nx))  # fixed the sampling of A to be between -1 and 1
                A /= np.max(np.abs(np.linalg.eigvals(A)))
                self.A = A * 0.95
                # A = np.zeros((nx, nx))
                # A[0,0] = 0.95
                # self.A = A

        if C_dist == "_gauss_C":
            normC = True
        else:
            normC = False
        if C_dist == "_single_system":
            self.C = np.array([
                [0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314, 0.57615733, 0.59204193, 0.57225191, 0.22308163, 0.95274901],
                [0.44712538, 0.84640867, 0.69947928, 0.29743695, 0.81379782, 0.39650574, 0.8811032, 0.58127287, 0.88173536, 0.69253159],
                [0.72525428, 0.50132438, 0.95608363, 0.6439902, 0.42385505, 0.60639321, 0.0191932, 0.30157482, 0.66017354, 0.29007761],
                [0.61801543, 0.4287687, 0.13547406, 0.29828233, 0.56996491, 0.59087276, 0.57432525, 0.65320082, 0.65210327, 0.43141844],
                [0.8965466, 0.36756187, 0.43586493, 0.89192336, 0.80619399, 0.70388858, 0.10022689, 0.91948261, 0.7142413, 0.99884701]
            ])
        else:
            self.C = np.eye(nx) if nx == ny else self.construct_C(self.A, ny, normC)

        self.S_state_inf = solve_ricc(self.A, np.eye(nx) * sigma_w ** 2)
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
        x0 = np.stack([
            np.random.multivariate_normal(np.zeros(nx), self.S_state_inf)
            for _ in range(batch_size)
        ])

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
    def construct_C(A, ny, normC): #normC is a boolean that determines if the C matrix is sampled from a normal distribution or a uniform distribution
        nx = A.shape[0]
        _O = [np.eye(nx)]
        for _ in range(nx - 1):
            _O.append(_O[-1] @ A)
        while True:
            if normC:
                C = np.random.normal(0, np.sqrt(0.333333333), (ny, nx))

                #scale C by the reciprocal of its frobenius norm
                # C = C/np.linalg.norm(C, ord='fro') #scale the matrix 
            else:
                C = np.random.rand(ny, nx) #uniform(0,1)
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


def _generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri=dataset_typ, C_dist=C_dist, n_noise=n_noise, new_eig = False)
    states, obs = fsim.simulate_steady(batch_size, n_positions)
    return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}

def _generate_lti_sample_new_eig(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri=dataset_typ, C_dist=C_dist, n_noise=n_noise, new_eig = True)
    states, obs = fsim.simulate_steady(batch_size, n_positions)
    return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}


def generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    while True:
        fsim, entry = _generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise)
        if check_validity(entry):
            return fsim, entry
        
def generate_lti_sample_new_eig(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    while True:
        fsim, entry = _generate_lti_sample_new_eig(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise)
        if check_validity(entry):
            return fsim, entry


####################################################################################################


def generate_changing_lti_sample(n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim1 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise, new_eig=False)
    fsim2 = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, n_noise=n_noise, new_eig=False)

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
