from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from collections import Counter


d = {'k':[[1,2],[2,3],[3,1]] , 'r': [[5,6],[7,5],[6,4]]}
new_feature = [6,3]
# [[plt.scatter(j[0],j[1],s=100,color=i) for j in d[i]] for i in d]
# plt.show()

def k_nearest_neighbor(data,predict,k=3):
    dist = []
    for grp in data:
        for feature in data[grp]:
            euleadian_dist = np.linalg.norm(np.array(feature) - np.array(predict))
            dist.append([euleadian_dist,grp])

    votes = [i[1] for i in sorted(dist)[:]]

    return Counter(votes).most_common(1)[0][0]

result = k_nearest_neighbor(d,new_feature,3)
print(result)

[[plt.scatter(j[0],j[1],s=100,color=i) for j in d[i]] for i in d]
plt.scatter(new_feature[0],new_feature[1],s=100,color=result,marker='*')
plt.show()