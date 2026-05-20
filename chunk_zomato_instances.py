import pickle
import math
import os

CHUNK_SIZE = 64 # 32, 64, 128, 256

with open('instances\zomato.pkl','rb') as f:
    data = pickle.load(f)

# Flatten all customer locations and demands
all_customers = []  # tuples (x,y,demand)
for depot, locs, demands, cap in data:
    for (x,y), d in zip(locs, demands):
        all_customers.append(((x,y), d))

# Create chunked instances
chunked = []
for i in range(0, len(all_customers), CHUNK_SIZE):
    chunk = all_customers[i:i+CHUNK_SIZE]
    locs = [list(c[0]) for c in chunk]
    demands = [int(c[1]) for c in chunk]
    # set depot as centroid
    xs = [p[0] for p in locs]
    ys = [p[1] for p in locs]
    depot = [sum(xs)/len(xs), sum(ys)/len(ys)]
    capacity = 10
    chunked.append((depot, locs, demands, capacity))

os.makedirs('instances', exist_ok=True)
with open('instances\zomato_chunked_64.pkl','wb') as f:
    pickle.dump(chunked, f)

print('created', len(chunked), 'chunked instances, each up to', CHUNK_SIZE, 'customers')
