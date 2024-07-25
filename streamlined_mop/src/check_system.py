# import numpy as np
import pickle

import numpy
import torch


# def convert_to_tensor_dicts(sim_objs):
#     tensor_dicts = []  # Initialize an empty list for dictionaries
#     for sim_obj in sim_objs:
#         # Convert .A and .C to tensors and create a dictionary
#         tensor_dict = {
#             'A': torch.from_numpy(sim_obj.A),
#             'C': torch.from_numpy(sim_obj.C)
#         }
#         tensor_dicts.append(tensor_dict)  # Append the dictionary to the list
#     return tensor_dicts


# # check systems that were trained and validated
# with open("../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C/data/val_upperTriA_gauss_C_sim_objs.pkl", "rb") as f:
#     sim_objs = pickle.load(f)

# print("type(sim_objs[0])", type(sim_objs[0]))
# # print the first system
# print("sim_objs[0].A", sim_objs[0].A)
# # check if the first system is upper triangular and print
# print("is upper triangular", numpy.allclose(sim_objs[0].A, numpy.triu(sim_objs[0].A)))
# # print just the diagonal elements of the first system
# print("diagonal elements", numpy.diagonal(sim_objs[0].A))
# print("\n\n\n")
# # do the above for the second third and fourth systems
# print("sim_objs[1].A", sim_objs[1].A)
# print("is upper triangular", numpy.allclose(sim_objs[1].A, numpy.triu(sim_objs[1].A)))
# print("diagonal elements", numpy.diagonal(sim_objs[1].A))
# print("\n\n\n")
# print("sim_objs[2].A", sim_objs[2].A)
# print("\nis upper triangular", numpy.allclose(sim_objs[2].A, numpy.triu(sim_objs[2].A)))
# print("\ndiagonal elements", numpy.diagonal(sim_objs[2].A))
# print("\neigenvalues", numpy.linalg.eigvals(sim_objs[2].A))

# # convert the systems to tensors
# tensor_dicts = convert_to_tensor_dicts(sim_objs)

# print("tensor_dicts", tensor_dicts)

# # save the tensors to a file in the same directory
# with open("../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C/data/val_upperTriA_gauss_C_sim_objs.pkl", "wb") as f:
#     pickle.dump(tensor_dicts, f)

# create a torch tensor of all zeros that is 3x3
A = torch.zeros(3, 3)

# take the pseudo-inverse of the tensor
A_pinv = torch.pinverse(A)

# print the pseudo-inverse
print("A_pinv", A_pinv)

#take the inverse of the tensor
A_inv = torch.inverse(A)

# print the inverse
print("A_inv", A_inv)

