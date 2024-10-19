import numpy as np

# data        : Training inputs  (num_samples x dim)
#we have 2-dimensional data points, let's say 10 of them
num_samples = 10
dim = 2

data = np.random.random((num_samples, dim))
print("Our data has shape {0} and looks like:\n".format(data.shape), data)

#the trick for incorporating bias in our matrices: add a column of ones
data = np.concatenate((np.ones((num_samples, 1)), data), axis=1)
print("Data after concatenation has shape {0} and looks like:\n".format(data.shape), data)

#transposing
print("Transposed data has shape {0} and looks like:\n".format(data.T.shape), data.T)

#demonstrating dot product
vector1 = np.array([1, 1.5, 0.6, 3.5])
vector2 = np.array([1, 2, 10, 2])

#the dot product should be 1*1 + 1.5*2 + 0.6*10 + 3.5*2 = 17
print("Dot product is: ", vector1.dot(vector2))

#demonstrating boolean indexing in numpy
indices = vector1 <= 1

print("Indices where vector1 is at most 1: ", indices)
#using them to change values
vector1[indices] = -1
print("vector1 is now\n", vector1)