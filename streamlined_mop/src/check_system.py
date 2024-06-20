# import numpy as np
import dyn_models
import pickle
import numpy
#check systems that were trained and validated
with open("/Users/sultandaniels/Documents/Transformer_Kalman/outputs/GPT2/240618_101949.96f4c1_upperTriA_unif_C/data/train_upperTriA_unif_C_sim_objs.pkl", "rb") as f:
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
print("is upper triangular", numpy.allclose(sim_objs[2].A, numpy.triu(sim_objs[2].A)))
print("diagonal elements", numpy.diagonal(sim_objs[2].A))
print("\n\n\n")
print("sim_objs[3].A", sim_objs[3].A)
print("is upper triangular", numpy.allclose(sim_objs[3].A, numpy.triu(sim_objs[3].A)))
print("diagonal elements", numpy.diagonal(sim_objs[3].A))

