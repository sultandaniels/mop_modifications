import pickle

#load the data pkl file from the data folder and print A and C
with open(f"../data/numpy/systems.pkl", "rb") as f:
   data = pickle.load(f)
   print("data.keys():", data.keys())
   print("data[A]:", data["F"])
   # print("data[C]:", data["C"])

#load test_sim.pt from data folder and print the A matrix
import torch
fsim = torch.load(f"../data/test_sim.pt")
print("fsim.A:", fsim[0].A)

