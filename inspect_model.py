import torch
m = torch.load('trained_models/cvrp/XE_1/model_incumbent_vrpRl13_C_XE_1_P_0.1_83390.pt', map_location='cpu')
print('keys:', list(m.keys()))
print('destroy_operation:', m.get('destroy_operation'))
print('p_destruction:', m.get('p_destruction'))
