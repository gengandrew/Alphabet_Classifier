from scipy.stats import multivariate_normal
from emnist import extract_training_samples
import matplotlib.pyplot as plt
import numpy as np
import sympy
import math

def getMaxConfidence(confidences):
    high = (0,0)
    mid = (0,0)
    low = (0,0)
    for confidence in confidences:
        if confidence[1] > high[1]:
            high = confidence
    
    for confidence in confidences:
        if confidence[1] > mid[1] and confidence[1] != high[1]:
            mid = confidence
    
    for confidence in confidences:
        if confidence[1] > low[1] and confidence[1] != high[1] and confidence[1] != mid[1]:
            low = confidence
    
    return (high, mid, low)


def getZeroEigvals(covars):
    eigvals = np.linalg.eigvals(covars[i])
    count = 0
    for eigval in eigvals:
        if eigval <= 0:
            count = count + 1
    
    return count


def graph(image):
    plt.figure(1, figsize=(3, 3))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


bestscore = 0
trainingSize = 5000
epochSize = 500
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'p', 'w', 'x', 'y', 'z']
images, labels = extract_training_samples('letters')

print("Full training set is: " + str(images.shape))

size = 0
parsedSet = []
for image in images:
    # Creating 7x7 Matrix
    matrix7 = np.zeros((7,7))
    for i in range(0,7):
        for j in range(0,7):
            element = 0
            for x in range(0,4):
                for y in range(0,4):
                    element = element + (image[i+x][j+y]/255)

            matrix7[i,j] = element/16
    
    # Creating 4x4 Matrix
    matrix4 = np.zeros((4,4))
    for i in range(0,4):
        for j in range(0,4):
            element = 0
            for x in range(0,7):
                for y in range(0,7):
                    element = element + (image[i+x][j+y]/255)

            matrix4[i,j] = element/49

    parsedSet.append(np.append(matrix7.flatten(),matrix4.flatten()))

    if size > trainingSize:
        break
    else:
        size = size + 1

# trainingSize = 0
# parsedSet = []
# for image in images:
#     # Creating 7x7 Matrix
#     matrix14 = np.zeros((14,14))
#     for i in range(0,14):
#         for j in range(0,14):
#             element = 0
#             for x in range(0,2):
#                 for y in range(0,2):
#                     element = element + (image[i+x][j+y]/255)

#             matrix14[i,j] = element/4

#     parsedSet.append(matrix14.flatten())
    
#     if trainingSize > 5000:
#         break
#     else:
#         trainingSize = trainingSize + 1

print("Number of training set taken is " + str(len(parsedSet)) + " with shape " + str(parsedSet[0].shape))
xtrains = parsedSet[1000:len(parsedSet)]
xtests = parsedSet[:1000]

for epoch in range(0,epochSize):
    means = [None]*len(alphabet)
    counts = [None]*len(alphabet)
    for i in range(0,len(alphabet)):
        means[i] = np.zeros(xtrains[0].shape)
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
        covars[i] = np.zeros((xtrains[0].shape[0],xtrains[0].shape[0]))
        counts[i] = 0

    # print("Starting to Calculate Covar")
    index = 0
    for xtrain in xtrains:
        label = labels[index]-1
        xNorm = xtrain - means[label]
        covars[label] = covars[label] + np.outer(xNorm,xNorm)
        counts[label] = counts[label]+1
        index = index+1

    for i in range(0,len(alphabet)):
        covars[i] = covars[i]/counts[i]

    print(covars[0].shape)
    for i in range(0,len(alphabet)):
        # _, inds = sympy.Matrix(covars[i]).T.rref()
        iteration = 0
        while (not np.all(np.linalg.eigvals(covars[i]) > 0)) or (abs(np.linalg.det(covars[i])) <= 0):
            fuzzScalar = np.random.normal(0,0.0001,1)[0]
            fuzz = fuzzScalar * np.identity(covars[0].shape[0])
            # fuzz = np.random.normal(0,0.01,covars[0].shape)
            covars[i] = covars[i] + fuzz
            iteration = iteration + 1

    # print("Starting to calculate the MVN density")
    scalars = [None]*len(alphabet)
    for i in range(0,len(alphabet)):
        scalars[i] = 1 / np.sqrt( pow((2*math.pi),56) * abs(np.linalg.det(covars[i])))

    epochscore = 0
    for curr in range(0,1000):
        confidences = [None]*len(alphabet)
        for i in range(0,len(alphabet)):
            confidences[i] = (i, multivariate_normal.pdf(xtests[curr], mean=means[i], cov=covars[i]))

        prediction = getMaxConfidence(confidences)
        if prediction[0][0] == labels[curr]-1:
            epochscore = epochscore + 1
        elif prediction[1][0] == labels[curr]-1:
            epochscore = epochscore + 0.5
        elif prediction[2][0] == labels[curr]-1:
            epochscore = epochscore + 0.25
    
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
