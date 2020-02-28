from emnist import extract_training_samples
import matplotlib.pyplot as plt
import numpy as np
import math

def getDensity(mean, covar, xtest):
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


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'p', 'w', 'x', 'y', 'z']
images, labels = extract_training_samples('letters')

print("Full training set is: " + str(images.shape))

# plt.figure(1, figsize=(3, 3))
# plt.imshow(images[29], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()

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
    
    if trainingSize > 2000:
        break
    else:
        trainingSize = trainingSize + 1


print("Number of training set taken is " + str(len(parsedSet)) + " with shape " + str(parsedSet[0].shape))
xtrains = parsedSet[500:len(parsedSet)]
xtests = parsedSet[:len(parsedSet)-1500]

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
    for row in range(0,56):
        for col in range(0,56):
            fuzz = np.random.normal(0,0.0001,1)[0]
            covars[i][row][col] = covars[i][row][col]

print("Starting to calculate the MVN density")

scalars = [None]*len(alphabet)
for i in range(0,len(alphabet)):
    scalars[i] = 1 / np.sqrt( pow((2*math.pi),56) * np.linalg.det(covars[i]))
    print(scalars[i])


# for test in range(0,500):
#     confidence = getDensity(mean, covar, xtests[test])
#     print("Confidence for " + str(test) + " is: " + str(confidence))

