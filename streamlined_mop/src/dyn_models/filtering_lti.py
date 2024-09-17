import numpy as np
import scipy as sc
from numpy import linalg as lin
from scipy import linalg as la
from filterpy.kalman import KalmanFilter
import random


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

def generate_random_list(lowest, highest):
    # Generate 8 random numbers between lowest and highest
    random_numbers = [random.uniform(lowest, highest) for _ in range(8)]
    
    # Add the specified lowest and highest numbers
    random_numbers.append(lowest)
    random_numbers.append(highest)
    
    #sort the list from lowest to highest
    random_numbers.sort(reverse=False)
    
    return random_numbers

def generate_random_rotation_matrix(n):
    # Generate a random 10x10 matrix
    random_matrix = np.random.randn(n, n)

    # Use QR decomposition to get a random rotation matrix
    Q, R = np.linalg.qr(random_matrix)
    return Q

def generate_random_mat_cond_number(n, cond_number):
    top = 1 #set the top bound for the singular values
    low = top / cond_number
    s = generate_random_list(low, top)
    S = np.diag(s)
    U = generate_random_rotation_matrix(n)
    VT = generate_random_rotation_matrix(n)
    random_matrix = U @ S @ VT

    # print("cond number = ", np.linalg.cond(random_matrix))
    return random_matrix

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
    def __init__(self, nx, ny, sigma_w, sigma_v, tri, C_dist, n_noise, new_eig, cond_num=10):
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v

        self.n_noise = n_noise

        if tri == "upperTriA":
            A = np.diag(np.random.uniform(-1, 1, nx)) * 0.95
            A[np.triu_indices(nx, 1)] = np.random.uniform(-1, 1, (nx ** 2 + nx) // 2 - nx)
            self.A = A
        # elif tri == "rotDiagA":
        elif tri.startswith("rotDiagA"):
            if tri == "rotDiagA":
                A = np.diag([0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9]) * 0.95 #generate a random diagonal matrix #values are handpicked to be near 1
            elif tri == "rotDiagA_unif":
                A = np.diag([-0.07679708, 0.42467112, 0.28797265, 0.08335881, -0.228187, 0.82329669, 0.75456837, 0.41401802, -0.93903797, -0.38618607]) #diagonal matrix, values are from unif(-0.95,0.95)
            elif tri == "rotDiagA_gauss":
                A = np.diag([0.63952862, 0.65871765, -0.02630632, 0.69056979, -0.23617829, -0.14567006,
                  -0.28050732, 0.8654012, -0.29325309, 0.95])

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
                [ 0.18456165,  0.16021711,  0.21647554,  0.26663666, -0.14021315, -0.11500664,
                0.22839415,  0.22999771,  0.24614115, -0.11526197],
                [ 0.16313339,  0.01252268,  0.37899637, -0.05354309, -0.0167731,  -0.20861132,
                -0.30752653,  0.03233317, -0.01655146, -0.07857788],
                [-0.10876597,  0.00779568,  0.13126843,  0.2767184,  -0.05360528,  0.15024604,
                -0.18804825,  0.32791277,  0.39922585, -0.1607608 ],
                [-0.27678485, -0.19300296, -0.15360622,  0.36045454, -0.12209954,  0.44985143,
                -0.46895359, -0.04361155,  0.04248527, -0.17745083],
                [ 0.10186719, -0.06536688, -0.30568725,  0.47224488, -0.36515259,  0.14956471,
                0.16592908,  0.11657638,  0.07751272, -0.05307784],
                [ 0.03795166,  0.37392155, -0.08363197, -0.43984755, -0.38351633,  0.22263573,
                0.21542389, -0.03819247, -0.04945852, -0.02149639],
                [ 0.0094122,   0.41767346,  0.2340716,   0.22193215,  0.47561425, -0.19203367,
                0.21373618,  0.10541387,  0.05244202,  0.46995048],
                [-0.03346005, -0.25898943, -0.01339142,  0.09552594,  0.03843111, -0.04369962,
                -0.01567509,  0.13728941,  0.39720152, -0.17611144],
                [-0.47235328, -0.26844127,  0.06685008,  0.5514679,   0.36689849, -0.01400152,
                -0.07387309, -0.1290582,  -0.34566337,  0.06974884],
                [-0.1204465,   0.26444027,  0.04800574, -0.1564598,  -0.2613636,   0.22308088,
                -0.13157398, -0.15901738, -0.14905257, -0.14572716]
            ]) #generate a random specific matrix
            A /= np.max(np.abs(np.linalg.eigvals(A)))
            self.A = A * 0.95 #scale the matrix
            print("A:", self.A)
            print("eigenvalues of A:", np.linalg.eigvals(self.A))

        elif tri == "cond_num":
            A = generate_random_mat_cond_number(nx, cond_num) #generate a random matrix with a condition number of cond_number
            self.A = A

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
                [ 1.58306722,  0.08254489,  0.63563991, -0.75589141,  0.45347395,  1.52444272,
                0.19327427, -1.45641797,  1.31582812, -0.54842482],
                [-0.35336291, -1.66812014,  0.68527397, -1.33242619, -0.74728405, -0.42868647,
                0.88728522,  1.31609117,  1.16449376, -1.34126255],
                [ 0.08083017,  0.66527164,  0.94678362, -1.35538693, -0.42848464,  0.30631194,
                1.07822856,  0.56396537,  1.00666173,  0.29519488],
                [-1.05025071, -0.72655103, -1.37426337, -1.33609509, -1.43163071, -0.78321476,
                0.57989037, -0.93342293, -1.21680371, -1.39550893],
                [ 0.34049993, -1.39124344,  1.17060005, -0.52646281, -1.23408133, -0.67153237,
                1.51040147,  0.91665462, -1.1607541, -0.10762072]
            ]) #generate a random specific matrix  
            print("C:", self.C)
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


def _generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1, cond_num=None):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri=dataset_typ, C_dist=C_dist, n_noise=n_noise, new_eig = False, cond_num=cond_num)
    states, obs = fsim.simulate_steady(batch_size, n_positions)
    return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}

def _generate_lti_sample_new_eig(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1):
    fsim = FilterSim(nx, ny, sigma_w, sigma_v, tri=dataset_typ, C_dist=C_dist, n_noise=n_noise, new_eig = True)
    states, obs = fsim.simulate_steady(batch_size, n_positions)
    return fsim, {"states": states, "obs": obs, "A": fsim.A, "C": fsim.C}


def generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=1, cond_num=None):
    while True:
        fsim, entry = _generate_lti_sample(C_dist, dataset_typ, batch_size, n_positions, nx, ny, sigma_w, sigma_v, n_noise=n_noise, cond_num=cond_num)
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
