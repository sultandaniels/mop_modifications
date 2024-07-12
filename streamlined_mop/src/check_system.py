# import numpy as np
import dyn_models
import pickle
import numpy
#check systems that were trained and validated
with open("/Users/sultandaniels/Documents/Transformer_Kalman/outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C/data/val_upperTriA_gauss_C_sim_objs.pkl", "rb") as f:
        sim_objs = pickle.load(f)

#print the first system
print("sim_objs[0].A", sim_objs[0].A)
#check if the first system is upper triangular and print
print("is upper triangular", numpy.allclose(sim_objs[0].A, numpy.triu(sim_objs[0].A)))
#print just the diagonal elements of the first system
print("diagonal elements", numpy.diagonal(sim_objs[0].A))
print("\n\n\n")
#do the above for the second third and fourth systems
print("sim_objs[1].A", sim_objs[1].A)
print("is upper triangular", numpy.allclose(sim_objs[1].A, numpy.triu(sim_objs[1].A)))
print("diagonal elements", numpy.diagonal(sim_objs[1].A))
print("\n\n\n")
print("sim_objs[2].A", sim_objs[2].A)
print("\nis upper triangular", numpy.allclose(sim_objs[2].A, numpy.triu(sim_objs[2].A)))
print("\ndiagonal elements", numpy.diagonal(sim_objs[2].A))
print("\neigenvalues", numpy.linalg.eigvals(sim_objs[2].A))