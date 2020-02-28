from scipy.stats import multivariate_normal
from emnist import extract_training_samples
import matplotlib.pyplot as plt
import numpy as np
import sympy
import math

def getMaxConfidence(confidences):
    max = (0,0)
    for confidence in confidences:
        if confidence[1] > max[1]:
            max = confidence
    
    return max

def getDensity(mean, covar, scalar, xtest):
    normX = xtest - mean
    iconvar = np.linalg.inv(covar)
    
    expres = np.dot(normX, iconvar)
    expres = np.dot(expres, normX)
    expres = -0.5*expres

    try:     
        expres = math.exp(expres)
    except OverflowError:
        expres = float('inf')

    confidence = scalar*expres
    return confidence


bestscore = 0
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'p', 'w', 'x', 'y', 'z']
images, labels = extract_training_samples('letters')

print("Full training set is: " + str(images.shape))

# trainingSize = 0
# parsedSet = []
# for image in images:
#     parsed = []
#     for i in range(0,28):
#         total = 0
#         for j in range(0,28):
#             total = total + (image[i][j]/255)
        
#         parsed.append(total/28)

#     for i in range(0, 28):
#         total = 0
#         for j in range(0,28):
#             total = total + (image[j][i]/255)
        
#         parsed.append(total/28)

#     parsedSet.append(np.array(parsed))
    
#     if trainingSize > 5000:
#         break
#     else:
#         trainingSize = trainingSize + 1

trainingSize = 0
parsedSet = []
for image in images:
    parsed = []
    for i in range(0,28):
        total = 0
        for j in range(0,28):
            total = total + (image[i][j]/255)
        
        parsed.append(total/28)

    for i in range(0, 28):
        total = 0
        for j in range(0,28):
            total = total + (image[j][i]/255)
        
        parsed.append(total/28)

    parsedSet.append(np.array(parsed))
    
    if trainingSize > 5000:
        break
    else:
        trainingSize = trainingSize + 1

print("Number of training set taken is " + str(len(parsedSet)) + " with shape " + str(parsedSet[0].shape))
xtrains = parsedSet[1000:len(parsedSet)]
xtests = parsedSet[:1000]

for epoch in range(0,500):
    means = [None]*len(alphabet)
    counts = [None]*len(alphabet)
    for i in range(0,len(alphabet)):
        means[i] = np.zeros((56,))
        counts[i] = 0

    index = 0
    for xtrain in xtrains:
        means[labels[index]-1] = means[labels[index]-1] + xtrain
        counts[labels[index]-1] = counts[labels[index]-1] + 1
        index = index + 1

    for i in range(0,len(alphabet)):
        means[i] = means[i]/counts[i]

    # print(len(means))
    # for i in range(0,len(alphabet)):
    #     print(means[i])

    covars = [None]*len(alphabet)
    counts = [None]*len(alphabet)
    for i in range(0,len(alphabet)):
        covars[i] = np.zeros((56,56))
        counts[i] = 0

    index = 0
    for xtrain in xtrains:
        label = labels[index]-1
        temp = xtrain - means[label]
        covars[label] = covars[label] + np.outer(temp,temp)
        counts[label] = counts[label]+1
        # if labels[index] == 1:
        #     temp = xtrain - mean
        #     covar = covar + np.outer(temp,temp)
        #     count = count+1
        index = index+1

    for i in range(0,len(alphabet)):
        covars[i] = covars[i]/counts[i]

    for i in range(0,len(alphabet)):
        # _, inds = sympy.Matrix(covars[i]).T.rref()
        # while not len(inds) == 56:
        for row in range(0,56):
            for col in range(0,56):
                fuzz = np.random.normal(0,0.0001,1)[0]
                covars[i][row][col] = covars[i][row][col] + fuzz
        
            # _, inds = sympy.Matrix(covars[i]).T.rref()
            # print("At stage " + str(i) + " and " + str(len(inds)))

    # print("Starting to calculate the MVN density")

    scalars = [None]*len(alphabet)
    for i in range(0,len(alphabet)):
        scalars[i] = 1 / np.sqrt( pow((2*math.pi),56) * abs(np.linalg.det(covars[i])))
        # print(scalars[i])

    epochscore = 0
    for test in range(0,1000):
        confidences = [None]*len(alphabet)
        for i in range(0,len(alphabet)):
            confidences[i] = (i, getDensity(means[i], covars[i], scalars[i], xtests[0]))

        prediction = getMaxConfidence(confidences)
        if prediction[0] == labels[test]-1:
            epochscore = epochscore + 1

    print("Epoch #" + str(epoch) + " with final score of " + str(epochscore))
    if epochscore > bestscore:
        bestscore = epochscore
        covarfilename = "e" + str(epoch) + "s" + str(epochscore) + "[covars]" + ".txt"
        meanfilename = "e" + str(epoch) + "s" + str(epochscore) + "[means]" + ".txt"

        with open(covarfilename, 'w') as outfile:
            outfile.write("# Covariance Matrices with Score " + str(epochscore) + "\n")
            aindex = 0
            for covar in covars:
                outfile.write("# Covariance number " + str(aindex) + " letter " + str(alphabet[aindex]) + "\n")
                np.savetxt(outfile, covar)
                aindex = aindex + 1

        with open(meanfilename, 'w') as outfile:
            outfile.write("# Mean Matrices with Score " + str(epochscore) + "\n")
            aindex = 0
            for mean in means:
                outfile.write("# Mean number " + str(aindex) + " letter " + str(alphabet[aindex]) + "\n")
                np.savetxt(outfile, mean)
                aindex = aindex + 1
        

# # Code for getting data out of file
# new_data = np.loadtxt(covarfilename)
# new_data = new_data.reshape((26, 56, 56))
# assert(np.all(new_data == covars))

# new_data = np.loadtxt(meanfilename)
# new_data = new_data.reshape((26, 56))
# assert(np.all(new_data == means))

# plt.figure(1, figsize=(3, 3))
# plt.imshow(images[0], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()