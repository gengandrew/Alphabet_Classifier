from scipy.stats import multivariate_normal
from emnist import extract_training_samples
import matplotlib.pyplot as plt
import numpy as np
import sympy
import math

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'p', 'w', 'x', 'y', 'z']

covars = np.loadtxt("covars.txt")
covars = covars.reshape((26, 65, 65))
# print(covars)

means = np.loadtxt("means.txt")
means = means.reshape((26, 65))
# print(means)

# model = [{"label": 'a', "cov": [[]], "mean": []}, ... ]
values = []
for aindex in range(0,26):
    value = dict()
    value['label'] = alphabet[aindex]
    value['covar'] = covars[aindex].tolist()
    value['mean'] = means[aindex].tolist()
    values.append(value)

with open("model.txt", 'w') as outfile:
    outfile.write(str(values))

print(values)