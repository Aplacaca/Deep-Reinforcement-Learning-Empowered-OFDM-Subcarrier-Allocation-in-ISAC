import numpy as np
from env.ofdm_env import ISAC_BS
import random
from itertools import permutations

np.random.seed(777)
random.seed(777)

full_set_u = list(permutations(list(range(8)),8))
print(len(full_set_u))
full_set_d = list(permutations(list(range(8)),4))
print(len(full_set_d))

env = ISAC_BS(N=8,N_c=8,N_r=4)

optimal_u = None
optimal_d = None
max_value = 0.0
for u in full_set_u:
    for d in full_set_d:
        _SET = np.eye(env.N)
        env.U = _SET[u,:]
        env.D = _SET[d,:]
        SUM_R_P,SUM_C_P,SUM_R_c,SUM_MI_r,EE_C,EE_R = env.get_performance()
        value = (EE_C/(5*1e4)+EE_R/10.0)
        if value > max_value:
            max_value = value
            optimal_u = u
            optimal_d = d
print("optimal reward: ", max_value)
print("optimal u: ", optimal_u)
print("optimal d: ", optimal_d)