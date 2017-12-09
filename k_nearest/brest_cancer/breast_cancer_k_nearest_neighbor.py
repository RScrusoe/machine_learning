from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random


def k_nearest_neighbor(data, predict, k=3):
    dist = []
    for grp in data:
        for feature in data[grp]:
            euleadian_dist = np.linalg.norm(
                np.array(feature) - np.array(predict))
            dist.append([euleadian_dist, grp])

    votes = [i[1] for i in sorted(dist)[:]]
    return Counter(votes).most_common(1)[0][0]

d = pd.read_csv('breast-cancer-wisconsin.data')
# This drops id column
d.drop(['id'], 1, inplace=True)
# This replaces missing data marked as '?' with -99999
d.replace('?', -99999, inplace=True)

# This replaces string and any other to float in data
d = d.astype(float).values.tolist()
random.shuffle(d)

#test size
test_size = 0.5

#creating train_set and test_set from train_data and test_data
#class 2 : Benign
#class 4 : Malignant
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = d[:-int(test_size * len(d))]
test_data = d[-int(test_size * len(d)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])
# [[plt.scatter(j[0],j[1],s=100,color=i) for j in d[i]] for i in d]
# plt.scatter(new_feature[0],new_feature[1],s=100,color=result,marker='*')
# plt.show()

correct = 0
total = 0

#running k_nearsest_neighbor algorithm
for grp in test_set:
    for instance in test_set[grp]:
        result = k_nearest_neighbor(train_set, instance, k=5)
        if result == grp:
            correct += 1
        total += 1

print("Correct:", correct, "\tOut of:", total, "\tAccuracy:",correct/total)
