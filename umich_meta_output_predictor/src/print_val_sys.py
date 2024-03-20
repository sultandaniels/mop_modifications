import pickle

#load the fsim_val pkl file from the data folder and print fsim.A and fsim.C
with open(f"../data/val_ypred_fsim_val.pkl", "rb") as f:
   fsim = pickle.load(f)
   print("fsim.A:", fsim.A)
   print("fsim.C:", fsim.C)

