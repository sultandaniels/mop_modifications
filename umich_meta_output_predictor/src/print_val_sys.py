import pickle

#load the data pkl file from the data folder and print A and C
with open(f"../data/numpy_three_sys_C_gaussian/systems.pkl", "rb") as f:
   data = pickle.load(f)
   print("data.keys():", data.keys())
   print("data[A]:", data["F"])
   # print("data[C]:", data["C"])

#load test_sim.pt from data folder and print the A matrix
import torch
fsim = torch.load(f"../data/numpy_three_sys_C_gaussian/test_sim.pt")
print("len(fsim):", len(fsim))
print("fsim.A:", fsim[0].A)
print("fsim.C:", fsim[0].C)
print("fsim.A:", fsim[1].A)
print("fsim.C:", fsim[1].C)
print("fsim.A:", fsim[2].A)
print("fsim.C:", fsim[2].C)

