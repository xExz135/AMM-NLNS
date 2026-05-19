import pickle
import numpy as np
from vrp.vrp_problem import VRPInstance

with open('instances\zomato.pkl','rb') as f:
    data = pickle.load(f)

bad = []
for i, inst in enumerate(data):
    locations = [inst[0]] + inst[1]
    demand = [0] + inst[2]
    instance = VRPInstance(len(locations)-1, np.array(locations), np.array(locations), np.array(demand), inst[3])
    instance.create_initial_solution()
    instance.destroy_point_based(1.0)
    if not instance.incomplete_tours:
        bad.append(i)

print('checked', len(data), 'bad', len(bad))
if len(bad) <= 50:
    print('bad indices:', bad)
else:
    print('First 50 bad indices:', bad[:50])
